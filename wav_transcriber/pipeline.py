"""
Core pipeline: separate speakers -> transcribe (any language, auto-detected) ->
score each speaker as human vs AI-generated/bot voice -> sentence-split the
human customer's speech.

Two ways speakers get separated, chosen automatically per file:
  - Channel split: if the WAV is stereo and the two channels carry genuinely
    different audio (the common call-recording layout: one speaker per
    channel), each channel is transcribed independently. No diarization
    model or Hugging Face token needed for these files.
  - Diarization: for mono or genuinely mixed-down audio, pyannote is used to
    figure out who spoke when. Requires a Hugging Face token.

Models used:
  - pyannote/speaker-diarization-community-1  (who spoke when, mixed-audio fallback)
  - openai-whisper                    (what was said, multilingual auto-detect)
  - garystafford/wav2vec2-deepfake-voice-detector (human vs AI-generated voice)

The AI-voice classifier is best-effort: it was fine-tuned on a specific set of
TTS engines and will not catch every synthetic voice. Use --ai-threshold to
tune sensitivity, and --customer to override its decision manually.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
AI_VOICE_MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
CLASSIFIER_SAMPLE_RATE = 16000
CLASSIFIER_CHUNK_SECONDS = 10.0
CHANNEL_CORRELATION_THRESHOLD = 0.95

# Short language codes (as Whisper detects, or as offered in a UI) -> NLLB-200 FLORES codes.
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "de": "deu_Latn",
    "id": "ind_Latn",
    "tl": "tgl_Latn",
    "bn": "ben_Beng",
    "ur": "urd_Arab",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sw": "swh_Latn",
}


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    wav_path: Path
    method: str  # "channel_split" or "diarization"
    language: str | None
    turns: list[Turn]
    speaker_ai_probability: dict[str, float]
    human_speakers: list[str]
    customer_speaker: str | None
    customer_sentences: list[str]
    full_transcript_path: Path
    customer_sentences_path: Path | None
    customer_chunks_dir: Path | None = None
    customer_chunks_manifest_path: Path | None = None
    used_second_speaker_heuristic: bool = False


@functools.lru_cache(maxsize=1)
def _load_diarization_pipeline(hf_token: str):
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_ID, token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


FASTER_WHISPER_MODEL_ALIASES = {"large": "large-v3"}


@functools.lru_cache(maxsize=4)
def _load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    model_size = FASTER_WHISPER_MODEL_ALIASES.get(model_size, model_size)
    # int8 on CPU: 4-5x faster than the original openai-whisper implementation at
    # equal-or-better accuracy, with negligible quality loss from quantization.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _run_whisper(model, audio) -> tuple[list[dict], str | None]:
    """Run faster-whisper and normalize its segment generator into the list-of-dict
    shape the rest of the pipeline expects. vad_filter uses a built-in speech
    detector so quiet/short utterances are caught rather than silently dropped, and
    a short min_silence_duration keeps turn boundaries responsive to real pauses.
    condition_on_previous_text=False avoids a known Whisper failure mode on noisy
    telephony audio where a poorly-decoded segment biases later segments into a
    repetition loop (e.g. "Clean Intimation Clean Intimation Clean Intimation...")."""
    segments_iter, info = model.transcribe(
        audio,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments_iter]
    return segments, info.language


@functools.lru_cache(maxsize=1)
def _load_ai_voice_classifier():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("audio-classification", model=AI_VOICE_MODEL_ID)


@functools.lru_cache(maxsize=1)
def _load_translation_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID)
    return tokenizer, model


def translate_sentences(sentences: list[str], source_lang: str | None, target_lang: str) -> list[str]:
    """Machine-translate sentences into target_lang (a short code like 'en', 'hi', 'es', ...)
    using NLLB-200, run locally. Returns sentences unchanged if the source and target already
    match, or if either language code isn't in NLLB_LANG_CODES."""
    if not sentences or not target_lang:
        return sentences
    if source_lang and source_lang.lower() == target_lang.lower():
        return sentences

    src_code = NLLB_LANG_CODES.get((source_lang or "en").lower(), "eng_Latn")
    tgt_code = NLLB_LANG_CODES.get(target_lang.lower())
    if not tgt_code:
        return sentences

    import torch

    tokenizer, model = _load_translation_model()
    tokenizer.src_lang = src_code
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.no_grad():
        generated = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def _ensure_punkt() -> None:
    import nltk

    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return
        except LookupError:
            continue
    nltk.download("punkt_tab", quiet=True)


# --------------------------------------------------------------------------
# Stereo channel-split path
# --------------------------------------------------------------------------


def _load_stereo_channels(wav_path: str):
    import soundfile as sf

    data, sr = sf.read(wav_path, always_2d=True)
    if data.shape[1] < 2:
        return None
    left = data[:, 0].astype(np.float32)
    right = data[:, 1].astype(np.float32)
    return left, right, sr


def channels_are_separated(wav_path: str, corr_threshold: float = CHANNEL_CORRELATION_THRESHOLD) -> bool:
    """True if the WAV is stereo and the two channels carry meaningfully different
    audio (typical of channel-separated call recordings, e.g. agent on one
    channel, customer on the other)."""
    channels = _load_stereo_channels(wav_path)
    if channels is None:
        return False
    left, right, _ = channels
    if np.std(left) < 1e-4 or np.std(right) < 1e-4:
        return False  # one channel is silent -> not usefully separated
    correlation = np.corrcoef(left, right)[0, 1]
    if np.isnan(correlation):
        return False
    return bool(correlation < corr_threshold)


def _resample_to_classifier_rate(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    if sr == CLASSIFIER_SAMPLE_RATE:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=CLASSIFIER_SAMPLE_RATE)


def _turns_from_whisper_segments(segments: list[dict], speaker_label: str) -> list[Turn]:
    turns = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            turns.append(Turn(speaker=speaker_label, start=seg["start"], end=seg["end"], text=text))
    return turns


def _transcribe_from_channels(
    wav_path_p: Path,
    whisper_model: str,
    ai_threshold: float,
    customer_speaker: str | None,
    out_dir_path: Path,
    export_chunks: bool = True,
    translate_to: str | None = None,
) -> TranscriptResult:
    left_raw, right_raw, sr = _load_stereo_channels(str(wav_path_p))
    left16 = _resample_to_classifier_rate(left_raw, sr)
    right16 = _resample_to_classifier_rate(right_raw, sr)

    model = _load_whisper_model(whisper_model)
    left_segments, left_lang = _run_whisper(model, left16)
    right_segments, right_lang = _run_whisper(model, right16)
    language = left_lang or right_lang

    turns = _turns_from_whisper_segments(left_segments, "CHANNEL_LEFT") + _turns_from_whisper_segments(
        right_segments, "CHANNEL_RIGHT"
    )
    turns.sort(key=lambda t: t.start)

    classifier = _load_ai_voice_classifier()
    speaker_ai_probability = {
        "CHANNEL_LEFT": _classify_clip(left16, CLASSIFIER_SAMPLE_RATE, classifier),
        "CHANNEL_RIGHT": _classify_clip(right16, CLASSIFIER_SAMPLE_RATE, classifier),
    }

    source_audio_by_speaker = {
        "CHANNEL_LEFT": (left16, CLASSIFIER_SAMPLE_RATE),
        "CHANNEL_RIGHT": (right16, CLASSIFIER_SAMPLE_RATE),
    }

    return _finalize_result(
        wav_path_p, "channel_split", language, turns, speaker_ai_probability,
        ai_threshold, customer_speaker, out_dir_path,
        export_chunks=export_chunks, source_audio_by_speaker=source_audio_by_speaker,
        translate_to=translate_to,
    )


# --------------------------------------------------------------------------
# Diarization path (mono / genuinely mixed-down audio)
# --------------------------------------------------------------------------


def diarize(wav_path: str, hf_token: str) -> list[tuple[float, float, str]]:
    pipeline = _load_diarization_pipeline(hf_token)
    output = pipeline(wav_path)
    # pyannote.audio 4.x wraps the Annotation in a DiarizeOutput; older versions
    # returned the Annotation directly. Handle both.
    annotation = getattr(output, "speaker_diarization", output)
    segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s[0])
    return segments


def transcribe_whisper(wav_path: str, model_size: str) -> tuple[list[dict], str | None]:
    model = _load_whisper_model(model_size)
    return _run_whisper(model, wav_path)


def _assign_speaker(seg_start: float, seg_end: float, diar_segments) -> str:
    best_speaker, best_overlap = None, 0.0
    for d_start, d_end, speaker in diar_segments:
        overlap = min(seg_end, d_end) - max(seg_start, d_start)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, speaker
    return best_speaker or "UNKNOWN"


def group_by_speaker(whisper_segments: list[dict], diar_segments) -> list[Turn]:
    turns: list[Turn] = []
    for seg in whisper_segments:
        speaker = _assign_speaker(seg["start"], seg["end"], diar_segments)
        text = seg["text"].strip()
        if turns and turns[-1].speaker == speaker:
            turns[-1].text += " " + text
            turns[-1].end = seg["end"]
        else:
            turns.append(Turn(speaker=speaker, start=seg["start"], end=seg["end"], text=text))
    return turns


def score_speakers_ai_probability(wav_path: str, turns: list[Turn]) -> dict[str, float]:
    """Average P(AI-generated) per speaker, computed over that speaker's own audio."""
    import librosa

    classifier = _load_ai_voice_classifier()
    y_full, sr = librosa.load(wav_path, sr=CLASSIFIER_SAMPLE_RATE, mono=True)

    per_speaker_probs: dict[str, list[float]] = {}
    for turn in turns:
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        clip = y_full[start_sample:end_sample]
        prob = _classify_clip(clip, sr, classifier)
        per_speaker_probs.setdefault(turn.speaker, []).append(prob)

    return {speaker: float(np.mean(probs)) for speaker, probs in per_speaker_probs.items()}


def _transcribe_from_diarization(
    wav_path_p: Path,
    hf_token: str,
    whisper_model: str,
    ai_threshold: float,
    customer_speaker: str | None,
    out_dir_path: Path,
    export_chunks: bool = True,
    translate_to: str | None = None,
) -> TranscriptResult:
    import librosa

    diar_segments = diarize(str(wav_path_p), hf_token)
    whisper_segments, language = transcribe_whisper(str(wav_path_p), whisper_model)
    turns = group_by_speaker(whisper_segments, diar_segments)
    speaker_ai_probability = score_speakers_ai_probability(str(wav_path_p), turns)

    y_full, sr = librosa.load(str(wav_path_p), sr=CLASSIFIER_SAMPLE_RATE, mono=True)
    source_audio_by_speaker = {speaker: (y_full, sr) for speaker in speaker_ai_probability}

    return _finalize_result(
        wav_path_p, "diarization", language, turns, speaker_ai_probability,
        ai_threshold, customer_speaker, out_dir_path,
        export_chunks=export_chunks, source_audio_by_speaker=source_audio_by_speaker,
        translate_to=translate_to,
    )


# --------------------------------------------------------------------------
# Shared: AI-voice classification, speaker picking, sentence splitting, output
# --------------------------------------------------------------------------


def _classify_clip(y: np.ndarray, sr: int, classifier) -> float:
    """Return P(AI-generated) for one audio clip, chunking long clips and averaging."""
    if len(y) == 0:
        return 0.5
    chunk_len = int(CLASSIFIER_CHUNK_SECONDS * sr)
    chunks = [y[i : i + chunk_len] for i in range(0, len(y), chunk_len)] or [y]
    fake_probs = []
    for chunk in chunks:
        if len(chunk) < sr * 0.3:
            continue
        scores = classifier({"array": chunk, "sampling_rate": sr})
        prob = next(
            (s["score"] for s in scores if "fake" in s["label"].lower() or s["label"] == "LABEL_1"),
            None,
        )
        if prob is None:
            prob = max((s["score"] for s in scores), default=0.5)
        fake_probs.append(prob)
    return float(np.mean(fake_probs)) if fake_probs else 0.5


def pick_customer_speaker(
    speaker_ai_probability: dict[str, float],
    ai_threshold: float = 0.5,
    forced_customer: str | None = None,
) -> tuple[str | None, list[str]]:
    """
    Returns (customer_speaker, human_speakers).
    If exactly one human speaker is found, it's auto-picked as the customer.
    If more than one, customer_speaker is None unless forced_customer is set —
    caller should re-run with an explicit choice.

    If diarization found exactly one speaker in the whole recording, they're picked
    as the customer regardless of the AI-voice classifier's verdict. The classifier
    is known-unreliable on narrowband/telephony audio (confirmed: genuinely human
    audio has scored >0.9 "AI" on real test files) — when there's only one voice in
    the entire file, silently dropping it to zero output because of one shaky
    classifier score is worse than trusting that the sole speaker is the one we want.
    """
    human_speakers = [s for s, p in speaker_ai_probability.items() if p < ai_threshold]

    if forced_customer:
        return forced_customer, human_speakers
    if len(human_speakers) == 1:
        return human_speakers[0], human_speakers
    if len(speaker_ai_probability) == 1:
        return next(iter(speaker_ai_probability)), human_speakers
    return None, human_speakers


def pick_second_speaker_as_customer(turns: list[Turn], human_speakers: list[str]) -> str | None:
    """
    Fallback for exactly two human speakers with no other way to tell them apart
    (mono/mixed audio, no channel convention): assume whoever speaks first is the
    agent (they usually greet first) and the second to speak is the customer.
    Only resolves the two-speaker case; three or more stays ambiguous.
    """
    if len(human_speakers) != 2:
        return None
    first_turn_time = {
        speaker: min((t.start for t in turns if t.speaker == speaker), default=float("inf"))
        for speaker in human_speakers
    }
    return max(human_speakers, key=lambda s: first_turn_time[s])


def export_speaker_chunks(
    source_audio: np.ndarray,
    sr: int,
    turns: list[Turn],
    speaker: str,
    out_dir_path: Path,
    stem: str,
) -> tuple[Path, Path, list[dict]]:
    """
    Write each of `speaker`'s turns as its own small WAV file (a single-turn
    chunk) plus a CSV manifest pairing each chunk with its transcribed text —
    usable directly as ground-truth samples for STT benchmarking.

    Returns (chunks_dir, manifest_path, manifest_rows).
    """
    import csv

    import soundfile as sf

    chunks_dir = out_dir_path / f"{stem}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    i = 0
    for t in turns:
        if t.speaker != speaker:
            continue
        i += 1
        start_sample = max(0, int(t.start * sr))
        end_sample = min(len(source_audio), int(t.end * sr))
        clip = source_audio[start_sample:end_sample]
        if len(clip) == 0:
            continue
        chunk_name = f"{stem}_turn{i:03d}_{t.start:.2f}-{t.end:.2f}.wav"
        chunk_path = chunks_dir / chunk_name
        sf.write(str(chunk_path), clip, sr)
        manifest_rows.append(
            {
                "chunk_file": str(chunk_path),
                "start": f"{t.start:.3f}",
                "end": f"{t.end:.3f}",
                "duration": f"{t.end - t.start:.3f}",
                "text": t.text,
            }
        )

    manifest_path = chunks_dir / f"{stem}_chunks_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_file", "start", "end", "duration", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return chunks_dir, manifest_path, manifest_rows


def sentences_for_speaker(turns: list[Turn], speaker: str) -> list[str]:
    """
    Split a speaker's speech into meaningful chunks. Turn boundaries (natural pauses
    in the audio, from Whisper's own segmentation) are the primary split — they always
    exist, regardless of punctuation. nltk further splits *within* a turn only when it
    contains more than one clear sentence. Joining everything into one block first and
    relying solely on punctuation to re-split it fails when the transcript has weak or
    no terminal punctuation (common in parts of STT output, and worse across languages).
    """
    _ensure_punkt()
    from nltk.tokenize import sent_tokenize

    sentences: list[str] = []
    for t in turns:
        if t.speaker != speaker:
            continue
        text = t.text.strip()
        if not text:
            continue
        parts = [p.strip() for p in sent_tokenize(text) if p.strip()]
        sentences.extend(parts or [text])
    return sentences


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _finalize_result(
    wav_path_p: Path,
    method: str,
    language: str | None,
    turns: list[Turn],
    speaker_ai_probability: dict[str, float],
    ai_threshold: float,
    customer_speaker: str | None,
    out_dir_path: Path,
    export_chunks: bool = True,
    source_audio_by_speaker: dict[str, tuple[np.ndarray, int]] | None = None,
    translate_to: str | None = None,
) -> TranscriptResult:
    stem = wav_path_p.stem
    turns = sorted(turns, key=lambda t: t.start)

    chosen_customer, human_speakers = pick_customer_speaker(
        speaker_ai_probability, ai_threshold=ai_threshold, forced_customer=customer_speaker
    )

    used_second_speaker_heuristic = False
    if not chosen_customer and len(human_speakers) > 1:
        heuristic_pick = pick_second_speaker_as_customer(turns, human_speakers)
        if heuristic_pick:
            chosen_customer = heuristic_pick
            used_second_speaker_heuristic = True

    full_transcript_path = out_dir_path / f"{stem}_full_transcript.txt"
    with open(full_transcript_path, "w") as f:
        f.write(f"# method: {method}\n")
        f.write(f"# language: {language}\n")
        for speaker, prob in sorted(speaker_ai_probability.items()):
            tag = "AI/bot" if prob >= ai_threshold else "human"
            f.write(f"# {speaker}: {tag} (P(AI)={prob:.2f})\n")
        if used_second_speaker_heuristic:
            f.write(
                f"# customer_speaker: {chosen_customer} "
                f"(heuristic: 2 human speakers found, assumed 2nd to speak is the customer)\n"
            )
        f.write("\n")
        for t in turns:
            f.write(f"[{fmt_time(t.start)} - {fmt_time(t.end)}] {t.speaker}: {t.text}\n")

    customer_sentences: list[str] = []
    customer_sentences_path = None
    customer_chunks_dir = None
    customer_chunks_manifest_path = None
    if chosen_customer:
        customer_sentences = sentences_for_speaker(turns, chosen_customer)
        # Translation always runs as a separate step on the customer-facing sentences
        # only — turns/chunks stay in the original spoken language, since chunk export
        # is meant as untranslated STT ground truth.
        if translate_to:
            customer_sentences = translate_sentences(customer_sentences, language, translate_to)
        customer_sentences_path = out_dir_path / f"{stem}_customer_sentences.txt"
        with open(customer_sentences_path, "w") as f:
            for s in customer_sentences:
                f.write(s + "\n")

        if export_chunks and source_audio_by_speaker and chosen_customer in source_audio_by_speaker:
            source_audio, sr = source_audio_by_speaker[chosen_customer]
            customer_chunks_dir, customer_chunks_manifest_path, _ = export_speaker_chunks(
                source_audio, sr, turns, chosen_customer, out_dir_path, stem
            )

    return TranscriptResult(
        wav_path=wav_path_p,
        method=method,
        language=language,
        turns=turns,
        speaker_ai_probability=speaker_ai_probability,
        human_speakers=human_speakers,
        customer_speaker=chosen_customer,
        customer_sentences=customer_sentences,
        full_transcript_path=full_transcript_path,
        customer_sentences_path=customer_sentences_path,
        customer_chunks_dir=customer_chunks_dir,
        customer_chunks_manifest_path=customer_chunks_manifest_path,
        used_second_speaker_heuristic=used_second_speaker_heuristic,
    )


def transcribe_customer(
    wav_path: str,
    hf_token: str | None = None,
    whisper_model: str = "medium",
    ai_threshold: float = 0.5,
    customer_speaker: str | None = None,
    customer_channel: str | None = "LEFT",
    out_dir: str | None = None,
    export_chunks: bool = True,
    translate_to: str | None = None,
) -> TranscriptResult:
    """
    End-to-end pipeline for one WAV file. Automatically picks the method:

      - Channel split (no hf_token needed) if the WAV is stereo with two
        genuinely different channels — each channel is transcribed on its own.
        The customer is picked by fixed channel convention (`customer_channel`,
        default "LEFT"), since that's a known, 100% reliable recording layout
        in most call-center pipelines — far more reliable than the AI-voice
        classifier alone, which struggles on narrowband/telephony audio. The
        classifier's score is still computed and reported for visibility.
      - Diarization (hf_token required) otherwise — pyannote figures out who
        spoke when in the mixed-down audio, and the AI-voice classifier picks
        the (single) human speaker as the customer.

    `customer_speaker` always wins if given (e.g. "CHANNEL_RIGHT" or a
    diarization label like "SPEAKER_01") — use it to override either method's
    default choice. If diarization finds more than one human speaker and
    neither `customer_speaker` nor a confident auto-pick applies,
    `customer_sentences` is empty and `human_speakers` lists the candidates so
    the caller can re-invoke with an explicit `customer_speaker`.

    If `export_chunks` is True (default), each of the customer's turns is also
    written out as its own small WAV file under `<stem>_chunks/`, paired with
    its transcribed text in `<stem>_chunks_manifest.csv` — ready to use as
    single-turn ground-truth samples for STT benchmarking. Chunk text always
    stays in the original spoken language, regardless of `translate_to` — it's
    meant as STT ground truth, which should reflect what was actually said.

    `translate_to` (a short code like "en", "hi", "es", ... — see
    NLLB_LANG_CODES) translates the customer's sentences into that language.
    "en" uses Whisper's own speech-to-English translation for best quality;
    any other code transcribes normally then runs a local NLLB-200 translation
    pass. Leave as None to keep the original spoken language.
    """
    wav_path_p = Path(wav_path).expanduser().resolve()
    if not wav_path_p.exists():
        raise FileNotFoundError(wav_path_p)

    hf_token = hf_token or os.environ.get("HF_TOKEN")

    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir else wav_path_p.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if channels_are_separated(str(wav_path_p)):
        effective_customer = customer_speaker or (
            f"CHANNEL_{customer_channel.upper()}" if customer_channel else None
        )
        return _transcribe_from_channels(
            wav_path_p, whisper_model, ai_threshold, effective_customer, out_dir_path,
            export_chunks=export_chunks, translate_to=translate_to,
        )

    if not hf_token:
        raise ValueError(
            "This file's channels aren't separated (mono, or a genuinely mixed-down "
            "stereo track), so speaker diarization is required. Pass hf_token "
            "(or set HF_TOKEN)."
        )
    return _transcribe_from_diarization(
        wav_path_p, hf_token, whisper_model, ai_threshold, customer_speaker, out_dir_path,
        export_chunks=export_chunks, translate_to=translate_to,
    )

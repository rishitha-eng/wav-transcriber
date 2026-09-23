import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from wav_transcriber.pipeline import (
    Turn,
    channels_are_separated,
    decide_customer_speaker,
    detect_bot_by_self_identification,
    export_speaker_chunks,
    fmt_time,
    group_by_speaker,
    pick_customer_speaker,
    pick_second_speaker_as_customer,
    sentences_for_speaker,
    translate_sentences,
)


def test_fmt_time():
    assert fmt_time(0) == "00:00:00"
    assert fmt_time(65) == "00:01:05"
    assert fmt_time(3661) == "01:01:01"


def test_group_by_speaker_merges_consecutive_same_speaker():
    whisper_segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "there"},
        {"start": 2.5, "end": 3.5, "text": "Hi"},
    ]
    diar_segments = [
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 4.0, "SPEAKER_01"),
    ]
    turns = group_by_speaker(whisper_segments, diar_segments)
    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "Hello there"
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[1].text == "Hi"


def test_pick_customer_speaker_single_human():
    scores = {"SPEAKER_00": 0.95, "SPEAKER_01": 0.10}
    customer, humans = pick_customer_speaker(scores, ai_threshold=0.5)
    assert customer == "SPEAKER_01"
    assert humans == ["SPEAKER_01"]


def test_pick_customer_speaker_multiple_humans_needs_manual_pick():
    scores = {"SPEAKER_00": 0.10, "SPEAKER_01": 0.20}
    customer, humans = pick_customer_speaker(scores, ai_threshold=0.5)
    assert customer is None
    assert set(humans) == {"SPEAKER_00", "SPEAKER_01"}


def test_pick_customer_speaker_forced_override():
    scores = {"SPEAKER_00": 0.10, "SPEAKER_01": 0.20}
    customer, humans = pick_customer_speaker(scores, ai_threshold=0.5, forced_customer="SPEAKER_00")
    assert customer == "SPEAKER_00"


def test_pick_second_speaker_as_customer_picks_later_talker():
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="Hi, thanks for calling."),
        Turn(speaker="SPEAKER_01", start=1.5, end=2.5, text="I have an issue."),
        Turn(speaker="SPEAKER_00", start=3.0, end=4.0, text="Sure, let me check."),
    ]
    result = pick_second_speaker_as_customer(turns, ["SPEAKER_00", "SPEAKER_01"])
    assert result == "SPEAKER_01"


def test_pick_second_speaker_as_customer_returns_none_for_three_or_more():
    turns = [
        Turn(speaker="A", start=0.0, end=1.0, text="x"),
        Turn(speaker="B", start=1.0, end=2.0, text="y"),
        Turn(speaker="C", start=2.0, end=3.0, text="z"),
    ]
    assert pick_second_speaker_as_customer(turns, ["A", "B", "C"]) is None


def test_translate_sentences_skips_when_source_equals_target():
    with patch("wav_transcriber.pipeline._load_translation_model") as mock_load:
        result = translate_sentences(["Hello."], "en", "en")
    assert result == ["Hello."]
    mock_load.assert_not_called()


def test_translate_sentences_skips_unknown_target_code():
    with patch("wav_transcriber.pipeline._load_translation_model") as mock_load:
        result = translate_sentences(["Hello."], "en", "xx-not-a-real-code")
    assert result == ["Hello."]
    mock_load.assert_not_called()


def test_translate_sentences_calls_model_with_correct_codes():
    import torch

    class FakeTokenizer:
        def __init__(self):
            self.src_lang = None

        def __call__(self, sentences, return_tensors="pt", padding=True):
            return {"input_ids": torch.zeros((len(sentences), 3), dtype=torch.long)}

        def convert_tokens_to_ids(self, token):
            self.requested_bos_token = token
            return 999

        def batch_decode(self, generated, skip_special_tokens=True):
            return [f"translated-{i}" for i in range(generated.shape[0])]

    class FakeModel:
        def generate(self, **kwargs):
            n = kwargs["input_ids"].shape[0]
            assert kwargs["forced_bos_token_id"] == 999
            return torch.zeros((n, 3), dtype=torch.long)

    fake_tokenizer = FakeTokenizer()
    with patch("wav_transcriber.pipeline._load_translation_model", return_value=(fake_tokenizer, FakeModel())):
        result = translate_sentences(["Hello.", "How are you?"], "en", "hi")

    assert result == ["translated-0", "translated-1"]
    assert fake_tokenizer.src_lang == "eng_Latn"
    assert fake_tokenizer.requested_bos_token == "hin_Deva"


def test_detect_bot_by_self_identification_finds_clear_disclosure():
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=3.0,
             text="Hi, it's Alex. I'm Alex, Heartland's virtual support assistant."),
        Turn(speaker="SPEAKER_01", start=3.5, end=4.0, text="I'd like to speak to a team member."),
    ]
    assert detect_bot_by_self_identification(turns) == "SPEAKER_00"


def test_detect_bot_by_self_identification_finds_hindi_disclosure():
    # Real case (file 3): a Hindi bot ("Riya") self-identifies with "service
    # champion" code-switched into English text — was missed entirely before this
    # phrase was added, causing the classifier fallback to label the bot as the
    # customer.
    turns = [
        Turn(speaker="SPEAKER_01", start=0.0, end=5.0,
             text="Good Afternoon, मैं रिया हूं, आपकी service champion, आज मैं आपकी मदद करूँगी"),
        Turn(speaker="SPEAKER_00", start=5.5, end=6.0, text="मुझे अपनी पॉलिसी चेक करनी है"),
    ]
    assert detect_bot_by_self_identification(turns) == "SPEAKER_01"


def test_detect_bot_by_self_identification_none_when_nobody_discloses():
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="Hi there, how can I help?"),
        Turn(speaker="SPEAKER_01", start=1.0, end=2.0, text="I have an issue with my order."),
    ]
    assert detect_bot_by_self_identification(turns) is None


def test_detect_bot_by_self_identification_none_when_ambiguous():
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="I'm an AI assistant."),
        Turn(speaker="SPEAKER_01", start=1.0, end=2.0, text="This is also a virtual agent."),
    ]
    assert detect_bot_by_self_identification(turns) is None


def test_decide_customer_speaker_content_wins_even_when_classifier_scores_both_as_ai():
    # Real case (file 1, "Champ"): diarization correctly separates bot and human,
    # but the acoustic classifier wrongly scores BOTH speakers as AI. Content-based
    # bot self-identification must still correctly resolve this.
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=6.0,
             text="Hello, I'm Champ, your AI powered assistant, here to help you."),
        Turn(speaker="SPEAKER_01", start=30.0, end=33.0, text="Yeah, I'm done."),
    ]
    scores = {"SPEAKER_00": 0.995, "SPEAKER_01": 0.918}  # both "AI" per classifier
    customer, human_speakers, used_heuristic = decide_customer_speaker(turns, scores, 0.5, None)
    assert customer == "SPEAKER_01"
    assert used_heuristic is False


def test_decide_customer_speaker_content_wins_even_when_classifier_scores_both_as_human():
    # Real case (file 4, "Benjamin"): classifier scored both speakers as human, and
    # the second-speaker-to-talk heuristic alone picked the wrong one (the bot's
    # later turn). Content-based detection must override both.
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=5.0,
             text="Hello there, this is Benjamin a virtual agent from oid calling on a recorded line."),
        Turn(speaker="SPEAKER_01", start=5.5, end=6.5, text="Yeah, this is Bess, right?"),
        Turn(speaker="SPEAKER_00", start=7.0, end=9.0, text="Just checking are you the policy owner?"),
        Turn(speaker="SPEAKER_01", start=9.5, end=10.0, text="Yes, I am."),
    ]
    scores = {"SPEAKER_00": 0.14, "SPEAKER_01": 0.01}  # both "human" per classifier
    customer, human_speakers, used_heuristic = decide_customer_speaker(turns, scores, 0.5, None)
    assert customer == "SPEAKER_01"
    assert used_heuristic is False


def test_decide_customer_speaker_falls_back_to_heuristic_without_content_signal():
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="Hi, thanks for calling."),
        Turn(speaker="SPEAKER_01", start=1.5, end=2.5, text="I have an issue."),
    ]
    scores = {"SPEAKER_00": 0.3, "SPEAKER_01": 0.3}  # both pass as human, indistinguishable
    customer, human_speakers, used_heuristic = decide_customer_speaker(turns, scores, 0.5, None)
    assert customer == "SPEAKER_01"
    assert used_heuristic is True


def test_decide_customer_speaker_single_overall_speaker_still_wins():
    turns = [Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="I'd like to speak to a team member.")]
    scores = {"SPEAKER_00": 0.94}
    customer, human_speakers, used_heuristic = decide_customer_speaker(turns, scores, 0.5, None)
    assert customer == "SPEAKER_00"


def test_decide_customer_speaker_forced_override_always_wins():
    turns = [
        Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="I'm a virtual assistant."),
        Turn(speaker="SPEAKER_01", start=1.0, end=2.0, text="Hi."),
    ]
    scores = {"SPEAKER_00": 0.1, "SPEAKER_01": 0.9}
    customer, human_speakers, used_heuristic = decide_customer_speaker(turns, scores, 0.5, "SPEAKER_00")
    assert customer == "SPEAKER_00"


def test_pick_customer_speaker_single_speaker_wins_even_if_scored_as_ai():
    # A lone speaker in the whole recording is picked as the customer regardless of
    # the AI-voice classifier's verdict — the classifier is known-unreliable on
    # narrowband audio, and dropping the only voice in the file is worse than
    # trusting it's the one we want (confirmed with real, verified-human test audio
    # that the classifier scored >0.9 "AI").
    scores = {"SPEAKER_00": 0.95}
    customer, humans = pick_customer_speaker(scores, ai_threshold=0.5)
    assert customer == "SPEAKER_00"
    assert humans == []


def test_pick_customer_speaker_no_humans_with_multiple_speakers():
    scores = {"SPEAKER_00": 0.95, "SPEAKER_01": 0.99}
    customer, humans = pick_customer_speaker(scores, ai_threshold=0.5)
    assert customer is None
    assert humans == []


def test_sentence_split_basic():
    from wav_transcriber.pipeline import _ensure_punkt

    _ensure_punkt()
    from nltk.tokenize import sent_tokenize

    text = "Hi there. I have an issue with my order. Can you help?"
    sentences = sent_tokenize(text)
    assert sentences == [
        "Hi there.",
        "I have an issue with my order.",
        "Can you help?",
    ]


def test_sentences_for_speaker_splits_on_turn_boundaries_without_punctuation():
    # No terminal punctuation at all - punctuation-only splitting would collapse
    # this into a single blob. Turn boundaries must still produce separate chunks.
    turns = [
        Turn(speaker="CUSTOMER", start=0.0, end=1.0, text="i want to check my order status"),
        Turn(speaker="BOT", start=1.0, end=2.0, text="sure one moment"),
        Turn(speaker="CUSTOMER", start=2.0, end=3.0, text="it has not arrived yet"),
    ]
    result = sentences_for_speaker(turns, "CUSTOMER")
    assert result == ["i want to check my order status", "it has not arrived yet"]


def test_sentences_for_speaker_splits_multiple_sentences_within_one_turn():
    turns = [
        Turn(
            speaker="CUSTOMER", start=0.0, end=3.0,
            text="Hi there. I have an issue with my order. Can you help?",
        ),
    ]
    result = sentences_for_speaker(turns, "CUSTOMER")
    assert result == ["Hi there.", "I have an issue with my order.", "Can you help?"]


def test_channels_are_separated_true_for_distinct_channels(tmp_path):
    import soundfile as sf

    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    left = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 660 * t).astype(np.float32)
    wav_path = tmp_path / "distinct.wav"
    sf.write(wav_path, np.stack([left, right], axis=1), sr)

    assert channels_are_separated(str(wav_path)) is True


def test_channels_are_separated_false_for_identical_channels(tmp_path):
    import soundfile as sf

    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    mono = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    wav_path = tmp_path / "identical.wav"
    sf.write(wav_path, np.stack([mono, mono], axis=1), sr)

    assert channels_are_separated(str(wav_path)) is False


def test_channels_are_separated_false_for_mono(tmp_path):
    import soundfile as sf

    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    mono = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    wav_path = tmp_path / "mono.wav"
    sf.write(wav_path, mono, sr)

    assert channels_are_separated(str(wav_path)) is False


def test_export_speaker_chunks_writes_one_wav_per_turn_and_manifest(tmp_path):
    sr = 8000
    duration_samples = sr * 5
    source_audio = np.sin(2 * np.pi * 220 * np.linspace(0, 5, duration_samples, endpoint=False)).astype(np.float32)

    turns = [
        Turn(speaker="CUSTOMER", start=0.0, end=1.0, text="Hi there."),
        Turn(speaker="BOT", start=1.0, end=2.0, text="I am the bot."),
        Turn(speaker="CUSTOMER", start=2.0, end=3.5, text="I have an issue."),
    ]

    chunks_dir, manifest_path, rows = export_speaker_chunks(
        source_audio, sr, turns, "CUSTOMER", tmp_path, "testcall"
    )

    assert chunks_dir.exists()
    wav_files = sorted(chunks_dir.glob("*.wav"))
    assert len(wav_files) == 2  # only CUSTOMER turns, not BOT
    assert manifest_path.exists()
    assert len(rows) == 2
    assert rows[0]["text"] == "Hi there."
    assert rows[1]["text"] == "I have an issue."

    import soundfile as sf

    data, chunk_sr = sf.read(str(wav_files[0]))
    assert chunk_sr == sr
    assert len(data) > 0


@pytest.mark.skipif(
    not (os.environ.get("WAV_TRANSCRIBER_TEST_AUDIO") and os.environ.get("HF_TOKEN")),
    reason="set WAV_TRANSCRIBER_TEST_AUDIO=/path/to/sample.wav and HF_TOKEN to run the full pipeline",
)
def test_end_to_end_pipeline_runs():
    from wav_transcriber.pipeline import transcribe_customer

    wav_path = os.environ["WAV_TRANSCRIBER_TEST_AUDIO"]
    hf_token = os.environ["HF_TOKEN"]
    result = transcribe_customer(wav_path, hf_token=hf_token, whisper_model="tiny")
    assert result.full_transcript_path.exists()
    assert len(result.turns) > 0

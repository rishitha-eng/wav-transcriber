"""CLI for the wav_transcriber pipeline: single-file or batch-folder mode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .pipeline import transcribe_customer


def _get_hf_token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("HF_TOKEN")


def _process_one(wav_path: Path, hf_token: str | None, args) -> None:
    print(f"\n=== {wav_path.name} ===")
    try:
        result = transcribe_customer(
            str(wav_path),
            hf_token=hf_token,
            whisper_model=args.model,
            ai_threshold=args.ai_threshold,
            customer_speaker=args.customer,
            customer_channel=args.customer_channel,
            out_dir=args.out_dir,
            export_chunks=not args.no_export_chunks,
            translate_to=args.translate_to,
        )
    except ValueError as e:
        print(f"  {e}")
        print(
            "  Set it with: export HF_TOKEN=hf_xxx   (or pass --hf-token hf_xxx)\n"
            "  Get one at https://huggingface.co/settings/tokens after accepting terms at\n"
            "  https://huggingface.co/pyannote/speaker-diarization-community-1"
        )
        return
    print(f"  Method: {result.method}")
    print(f"  Language detected: {result.language}")
    for speaker, prob in sorted(result.speaker_ai_probability.items()):
        tag = "AI/bot" if prob >= args.ai_threshold else "human"
        print(f"  {speaker}: {tag} (P(AI)={prob:.2f})")
    print(f"  Full transcript: {result.full_transcript_path}")

    if result.customer_speaker:
        print(
            f"  Customer speaker: {result.customer_speaker} "
            f"({len(result.customer_sentences)} sentences)"
        )
        print(f"  Customer sentences: {result.customer_sentences_path}")
        if result.customer_chunks_dir:
            print(f"  Customer turn chunks: {result.customer_chunks_dir}")
            print(f"  Chunk manifest (chunk_file,start,end,duration,text): {result.customer_chunks_manifest_path}")
    elif len(result.human_speakers) > 1:
        print(
            f"  Multiple human speakers found ({', '.join(result.human_speakers)}) — "
            f"re-run with --customer <label> to pick one."
        )
    else:
        print("  No human speaker detected above the AI threshold.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diarize + transcribe WAV call recordings (any language, auto-detected), "
            "auto-classify each speaker as human vs AI-generated/bot voice, sentence-split "
            "the human customer's speech, and export each of their turns as its own WAV "
            "chunk + text (ground-truth-ready for STT benchmarking)."
        )
    )
    parser.add_argument("path", help="A WAV file, or a folder of WAV files for batch mode")
    parser.add_argument("--customer", help="Force a specific speaker label as the customer (e.g. SPEAKER_01, CHANNEL_RIGHT) — overrides everything else")
    parser.add_argument(
        "--customer-channel",
        choices=["LEFT", "RIGHT"],
        default="LEFT",
        help=(
            "For channel-separated stereo files: which channel is the customer, by fixed "
            "recording convention (default: LEFT). Ignored if --customer is set, or for "
            "files that need diarization."
        ),
    )
    parser.add_argument(
        "--model", default="medium", help="Whisper model size: tiny/base/small/medium/large (default: medium)"
    )
    parser.add_argument(
        "--ai-threshold",
        type=float,
        default=0.5,
        help="P(AI) at/above which a speaker is treated as bot/synthetic (default: 0.5)",
    )
    parser.add_argument("--hf-token", help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: alongside each WAV file)")
    parser.add_argument(
        "--no-export-chunks",
        action="store_true",
        help="Skip writing each customer turn as its own WAV file + manifest (on by default)",
    )
    parser.add_argument(
        "--translate-to",
        default=None,
        help=(
            "Translate customer sentences into this language code (e.g. en, hi, es, fr, ar, "
            "pt, de, id, zh, ja, ru, ta, te, mr, gu, kn, ml, pa, vi, th, ...). "
            "Leave unset to keep the original spoken language."
        ),
    )
    args = parser.parse_args()

    hf_token = _get_hf_token(args.hf_token)
    if not hf_token:
        print(
            "No Hugging Face token set — files with channel-separated audio will still "
            "work. Files needing diarization will report what's missing when hit.\n"
        )
    input_path = Path(args.path).expanduser().resolve()

    if not input_path.exists():
        sys.exit(f"Path not found: {input_path}")

    if input_path.is_dir():
        wav_files = sorted(input_path.glob("*.wav"))
        if not wav_files:
            sys.exit(f"No .wav files found in {input_path}")
        print(f"Found {len(wav_files)} WAV file(s) to process.")
        succeeded = 0
        for wav_path in wav_files:
            try:
                _process_one(wav_path, hf_token, args)
                succeeded += 1
            except Exception as e:
                print(f"  ERROR processing {wav_path.name}: {e}")
        print(f"\nDone. Processed {succeeded}/{len(wav_files)} files successfully.")
    else:
        _process_one(input_path, hf_token, args)


if __name__ == "__main__":
    main()

"""
Stage 3: chunk each downloaded recording into single turns + generate ground-truth text
(reusing the wav_transcriber pipeline), then assemble a CSV + flat audio folder matching
stt-benchmarking-go's `process-csv` input format: (Filename, Transcript) columns, and a
folder of identically-named .wav files.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from wav_transcriber.pipeline import transcribe_customer


def build_ground_truth(
    download_manifest_rows: list[dict],
    recordings_dir: Path,
    output_dir: Path,
    whisper_model: str = "small",
    hf_token: str | None = None,
) -> list[dict]:
    """
    Runs transcribe_customer on each successfully-downloaded recording, flattens its
    per-turn chunks into <output_dir>/benchmark_audio/ (filenames prefixed with callsid
    to stay unique across recordings), and writes:
      - <output_dir>/benchmark_manifest.csv   (full detail: filename, text, callsid, region, language, confidence)
      - <output_dir>/process_csv_input.csv    (just Filename,Transcript - feed this straight into process-csv)

    Recordings that fail (bad download, unsupported audio, needs diarization but no
    hf_token, etc.) are logged and skipped rather than stopping the whole batch.
    """
    benchmark_audio_dir = output_dir / "benchmark_audio"
    benchmark_audio_dir.mkdir(parents=True, exist_ok=True)

    master_rows: list[dict] = []
    for row in download_manifest_rows:
        if str(row.get("download_success")).lower() != "true":
            continue
        callsid = row["callsid"]
        wav_path = recordings_dir / f"{callsid}.wav"
        if not wav_path.exists():
            continue

        try:
            result = transcribe_customer(
                str(wav_path),
                hf_token=hf_token,
                whisper_model=whisper_model,
                out_dir=str(output_dir / "per_call" / callsid),
                export_chunks=True,
            )
        except Exception as e:  # noqa: BLE001 - logged and skipped, doesn't stop the batch
            print(f"  Skipping {callsid}: {e}")
            continue

        if not result.customer_chunks_dir or not result.customer_chunks_manifest_path:
            continue

        with open(result.customer_chunks_manifest_path) as f:
            for chunk_row in csv.DictReader(f):
                original_name = Path(chunk_row["chunk_file"]).name
                flat_name = f"{callsid}_{original_name}"
                shutil.copy(result.customer_chunks_dir / original_name, benchmark_audio_dir / flat_name)
                master_rows.append(
                    {
                        "Filename": Path(flat_name).stem,
                        "Transcript": chunk_row["text"],
                        "callsid": callsid,
                        "region": row.get("region", ""),
                        "stt_language": row.get("stt_language", ""),
                        "stt_confidence": row.get("stt_confidence", ""),
                    }
                )

    master_csv_path = output_dir / "benchmark_manifest.csv"
    with open(master_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Filename", "Transcript", "callsid", "region", "stt_language", "stt_confidence"]
        )
        writer.writeheader()
        writer.writerows(master_rows)

    # process-csv only reads columns 0 and 1 and skips the header row.
    process_csv_ready_path = output_dir / "process_csv_input.csv"
    with open(process_csv_ready_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Transcript"])
        for r in master_rows:
            writer.writerow([r["Filename"], r["Transcript"]])

    return master_rows


def main() -> None:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--download-manifest", default=None, help="CSV from stage 2 (default: <output_dir>/download_manifest.csv)"
    )
    parser.add_argument("--model", default="small", help="Whisper model size (use medium/large for real ground truth)")
    parser.add_argument("--hf-token", default=None, help="HF token, needed only if any recordings require diarization")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.download_manifest) if args.download_manifest else cfg.output_dir / "download_manifest.csv"

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    recordings_dir = cfg.output_dir / "recordings"
    master_rows = build_ground_truth(
        rows, recordings_dir, cfg.output_dir, whisper_model=args.model, hf_token=args.hf_token
    )

    print(f"Generated {len(master_rows)} ground-truth chunks.")
    print(f"process-csv-ready file: {cfg.output_dir / 'process_csv_input.csv'}")
    print(f"Audio folder for --audio flag: {cfg.output_dir / 'benchmark_audio'}")


if __name__ == "__main__":
    main()

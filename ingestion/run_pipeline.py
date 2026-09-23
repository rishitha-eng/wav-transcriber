"""
Run the full ingestion pipeline end-to-end: OpenSearch query -> Azure download ->
chunk + ground truth -> process-csv-ready output.

Start with a small time range in config.json to validate the flow before scaling up
(per the plan: validate small, then widen).
"""

from __future__ import annotations

import argparse
import csv

from .azure_downloader import download_all
from .config import load_config
from .ground_truth import build_ground_truth
from .opensearch_client import collect_stratified_samples, make_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--model", default="small", help="Whisper model size for ground truth (use medium/large for real benchmarking)"
    )
    parser.add_argument("--hf-token", default=None, help="HF token, only needed if any recordings require diarization")
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: querying OpenSearch ===")
    client = make_client(cfg.opensearch)
    rows = collect_stratified_samples(client, cfg.opensearch, cfg.selection)
    callsids_csv = cfg.output_dir / "callsids.csv"
    with open(callsids_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["callsid", "region", "stt_language", "stt_confidence", "recording_link", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Found {len(rows)} candidates -> {callsids_csv}")

    print("\n=== Stage 2: downloading recordings from Azure ===")
    recordings_dir = cfg.output_dir / "recordings"
    results = download_all(cfg.azure, rows, recordings_dir, max_workers=args.max_workers)
    download_manifest_csv = cfg.output_dir / "download_manifest.csv"
    with open(download_manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "callsid", "region", "stt_language", "stt_confidence", "recording_link",
                "timestamp", "download_success", "download_info",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    succeeded = sum(1 for r in results if r["download_success"])
    print(f"Downloaded {succeeded}/{len(results)} -> {download_manifest_csv}")

    print("\n=== Stage 3: chunking + ground truth ===")
    master_rows = build_ground_truth(
        results, recordings_dir, cfg.output_dir, whisper_model=args.model, hf_token=args.hf_token
    )
    print(f"Generated {len(master_rows)} ground-truth chunks.")

    print("\nDone. Feed this into the benchmarking tool with:")
    print(
        f"  go run cmd/process-csv/main.go -csv {cfg.output_dir / 'process_csv_input.csv'} "
        f"-audio {cfg.output_dir / 'benchmark_audio'} -output report.xlsx -provider <provider>"
    )


if __name__ == "__main__":
    main()

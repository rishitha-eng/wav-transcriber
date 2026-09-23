"""Stage 2: bulk-download recordings referenced in callsids.csv from Azure Storage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import AzureConfig


def _download_direct_url(url: str, dest_path: Path) -> None:
    """Recording link is already a fully-signed, directly-downloadable URL."""
    import requests

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def _download_azure_blob(cfg: AzureConfig, blob_path: str, dest_path: Path) -> None:
    """Recording link is a blob path; use the configured Storage Account connection string."""
    from azure.storage.blob import BlobServiceClient

    service_client = BlobServiceClient.from_connection_string(cfg.connection_string)
    container_client = service_client.get_container_client(cfg.container_name)
    blob_client = container_client.get_blob_client(blob_path)
    with open(dest_path, "wb") as f:
        stream = blob_client.download_blob()
        stream.readinto(f)


def download_one(cfg: AzureConfig, callsid: str, recording_link: str, recordings_dir: Path) -> tuple[str, bool, str]:
    dest_path = recordings_dir / f"{callsid}.wav"
    try:
        if cfg.recording_link_is_direct_url:
            _download_direct_url(recording_link, dest_path)
        else:
            _download_azure_blob(cfg, recording_link, dest_path)
        return callsid, True, str(dest_path)
    except Exception as e:  # noqa: BLE001 - reported per-row in the manifest, not swallowed
        return callsid, False, str(e)


def download_all(cfg: AzureConfig, rows: list[dict], recordings_dir: Path, max_workers: int = 6) -> list[dict]:
    """Downloads concurrently; one failure doesn't stop the batch. Returns rows annotated
    with download_success/download_info, ready to write out as a manifest CSV."""
    recordings_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one, cfg, row["callsid"], row["recording_link"], recordings_dir): row
            for row in rows
            if row.get("callsid") and row.get("recording_link")
        }
        for future in as_completed(futures):
            row = futures[future]
            callsid, success, info = future.result()
            results.append({**row, "download_success": success, "download_info": info})
    return results


def main() -> None:
    import argparse
    import csv

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--callsids-csv", default=None, help="Input CSV from stage 1 (default: <output_dir>/callsids.csv)"
    )
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    callsids_csv = Path(args.callsids_csv) if args.callsids_csv else cfg.output_dir / "callsids.csv"

    with open(callsids_csv) as f:
        rows = list(csv.DictReader(f))

    recordings_dir = cfg.output_dir / "recordings"
    results = download_all(cfg.azure, rows, recordings_dir, max_workers=args.max_workers)

    manifest_path = cfg.output_dir / "download_manifest.csv"
    fieldnames = [
        "callsid", "region", "stt_language", "stt_confidence", "recording_link",
        "timestamp", "download_success", "download_info",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    succeeded = sum(1 for r in results if r["download_success"])
    print(f"Downloaded {succeeded}/{len(results)} recordings.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

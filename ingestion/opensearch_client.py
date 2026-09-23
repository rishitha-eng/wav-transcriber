"""
Stage 1: query OpenSearch for a stratified sample of callsids.

Strategy: discover every (region, stt_language) combination actually present in the
time range, then for each combination pull the lowest-STT-confidence recordings first
(per confidence_buckets in config) — so the sample set covers every language used across
every region, with low-confidence recordings over-represented, as requested.
"""

from __future__ import annotations

from .config import OpenSearchConfig, SelectionConfig


def make_client(cfg: OpenSearchConfig):
    from opensearchpy import OpenSearch

    http_auth = None
    if cfg.auth_method == "basic" and cfg.username:
        http_auth = (cfg.username, cfg.password)

    return OpenSearch(
        hosts=[{"host": cfg.host, "port": cfg.port}],
        http_auth=http_auth,
        use_ssl=cfg.use_ssl,
        verify_certs=cfg.verify_certs,
    )


def discover_region_language_pairs(client, cfg: OpenSearchConfig, sel: SelectionConfig) -> list[tuple[str, str]]:
    """Aggregate distinct (region, stt_language) combinations present in the time range."""
    query = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"range": {cfg.field_timestamp: {"gte": sel.start_time, "lte": sel.end_time}}},
                    {"terms": {cfg.field_region: sel.regions}},
                ]
            }
        },
        "aggs": {
            "by_region": {
                "terms": {"field": cfg.field_region, "size": 100},
                "aggs": {"by_language": {"terms": {"field": cfg.field_language, "size": 100}}},
            }
        },
    }
    resp = client.search(index=cfg.index, body=query)
    pairs = []
    for region_bucket in resp["aggregations"]["by_region"]["buckets"]:
        region = region_bucket["key"]
        for lang_bucket in region_bucket["by_language"]["buckets"]:
            pairs.append((region, lang_bucket["key"]))
    return pairs


def fetch_lowest_confidence_samples(
    client,
    cfg: OpenSearchConfig,
    sel: SelectionConfig,
    region: str,
    language: str,
    confidence_min: float,
    confidence_max: float,
    size: int,
) -> list[dict]:
    """Up to `size` docs for one (region, language, confidence bucket), lowest confidence first."""
    query = {
        "size": size,
        "query": {
            "bool": {
                "filter": [
                    {"range": {cfg.field_timestamp: {"gte": sel.start_time, "lte": sel.end_time}}},
                    {"term": {cfg.field_region: region}},
                    {"term": {cfg.field_language: language}},
                    {"range": {cfg.field_confidence: {"gte": confidence_min, "lt": confidence_max}}},
                ]
            }
        },
        "sort": [{cfg.field_confidence: "asc"}],
    }
    resp = client.search(index=cfg.index, body=query)
    rows = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        rows.append(
            {
                "callsid": src.get(cfg.field_callsid),
                "region": src.get(cfg.field_region),
                "stt_language": src.get(cfg.field_language),
                "stt_confidence": src.get(cfg.field_confidence),
                "recording_link": src.get(cfg.field_recording_link),
                "timestamp": src.get(cfg.field_timestamp),
            }
        )
    return rows


def collect_stratified_samples(client, cfg: OpenSearchConfig, sel: SelectionConfig) -> list[dict]:
    """Build the full sample set: every (region, language) pair found, lowest-confidence first."""
    pairs = discover_region_language_pairs(client, cfg, sel)
    all_rows: list[dict] = []
    seen_callsids: set[str] = set()
    for region, language in pairs:
        for lo, hi in sel.confidence_buckets:
            rows = fetch_lowest_confidence_samples(
                client, cfg, sel, region, language, lo, hi, sel.samples_per_region_language_bucket
            )
            for row in rows:
                if row["callsid"] and row["callsid"] not in seen_callsids:
                    seen_callsids.add(row["callsid"])
                    all_rows.append(row)
    return all_rows


def main() -> None:
    import argparse
    import csv

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--out", default=None, help="Output CSV path (default: <output_dir>/callsids.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = args.out
    from pathlib import Path

    out_path = Path(out_path) if out_path else cfg.output_dir / "callsids.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(cfg.opensearch)
    rows = collect_stratified_samples(client, cfg.opensearch, cfg.selection)

    fieldnames = ["callsid", "region", "stt_language", "stt_confidence", "recording_link", "timestamp"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} candidate recordings across regions/languages/confidence buckets.")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

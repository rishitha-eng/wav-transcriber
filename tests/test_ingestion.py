import csv
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.azure_downloader import download_all
from ingestion.config import AzureConfig, OpenSearchConfig, SelectionConfig
from ingestion.ground_truth import build_ground_truth
from ingestion.opensearch_client import collect_stratified_samples


def _os_cfg():
    return OpenSearchConfig(
        host="host", port=9200, use_ssl=True, verify_certs=True,
        auth_method="basic", username="u", password="p",
        index="voice-vss-logs-*",
        field_callsid="callsid", field_language="sttLanguage", field_confidence="sttConfidence",
        field_region="region", field_recording_link="recordingLink", field_timestamp="@timestamp",
    )


def _sel_cfg():
    return SelectionConfig(
        start_time="2026-09-01T00:00:00Z", end_time="2026-09-22T00:00:00Z",
        regions=["r0", "r2"], confidence_buckets=[(0.0, 0.5), (0.5, 1.01)],
        samples_per_region_language_bucket=2,
    )


class FakeOpenSearchClient:
    """Returns the aggregation response for discovery queries, and per-(region,language,
    confidence-bucket) hits for term queries — inspecting the query body to tell them apart."""

    def __init__(self, hits_by_key):
        self.hits_by_key = hits_by_key
        self.search_calls = []

    def search(self, index, body):
        self.search_calls.append(body)
        if body.get("size") == 0:  # discovery aggregation
            return {
                "aggregations": {
                    "by_region": {
                        "buckets": [
                            {"key": "r0", "by_language": {"buckets": [{"key": "en"}, {"key": "hi"}]}},
                            {"key": "r2", "by_language": {"buckets": [{"key": "ar"}]}},
                        ]
                    }
                }
            }
        filters = body["query"]["bool"]["filter"]
        region = next(f["term"]["region"] for f in filters if "term" in f and "region" in f["term"])
        language = next(f["term"]["sttLanguage"] for f in filters if "term" in f and "sttLanguage" in f["term"])
        conf_filter = next(f["range"]["sttConfidence"] for f in filters if "range" in f and "sttConfidence" in f["range"])
        key = (region, language, conf_filter["gte"], conf_filter["lt"])
        hits = self.hits_by_key.get(key, [])
        return {"hits": {"hits": [{"_source": h} for h in hits]}}


def test_collect_stratified_samples_covers_all_pairs_and_dedupes():
    hits_by_key = {
        ("r0", "en", 0.0, 0.5): [
            {"callsid": "c1", "region": "r0", "sttLanguage": "en", "sttConfidence": 0.1, "recordingLink": "url1"},
        ],
        ("r0", "hi", 0.0, 0.5): [
            {"callsid": "c2", "region": "r0", "sttLanguage": "hi", "sttConfidence": 0.2, "recordingLink": "url2"},
        ],
        ("r2", "ar", 0.5, 1.01): [
            {"callsid": "c3", "region": "r2", "sttLanguage": "ar", "sttConfidence": 0.9, "recordingLink": "url3"},
            # duplicate callsid should be deduped even if seen again
            {"callsid": "c3", "region": "r2", "sttLanguage": "ar", "sttConfidence": 0.9, "recordingLink": "url3"},
        ],
    }
    client = FakeOpenSearchClient(hits_by_key)
    rows = collect_stratified_samples(client, _os_cfg(), _sel_cfg())

    callsids = {r["callsid"] for r in rows}
    assert callsids == {"c1", "c2", "c3"}
    assert len(rows) == 3  # deduped despite c3 appearing twice
    langs_by_region = {(r["region"], r["stt_language"]) for r in rows}
    assert ("r0", "en") in langs_by_region
    assert ("r0", "hi") in langs_by_region
    assert ("r2", "ar") in langs_by_region


def test_download_all_handles_success_and_failure(tmp_path):
    rows = [
        {"callsid": "good", "recording_link": "http://example.com/good.wav", "region": "r0", "stt_language": "en"},
        {"callsid": "bad", "recording_link": "http://example.com/bad.wav", "region": "r0", "stt_language": "en"},
    ]

    def fake_download_direct_url(url, dest_path):
        if "bad" in url:
            raise RuntimeError("404 not found")
        dest_path.write_bytes(b"RIFF....fake wav bytes")

    with patch("ingestion.azure_downloader._download_direct_url", side_effect=fake_download_direct_url):
        cfg = AzureConfig(recording_link_is_direct_url=True, connection_string="", container_name="")
        results = download_all(cfg, rows, tmp_path, max_workers=2)

    by_callsid = {r["callsid"]: r for r in results}
    assert by_callsid["good"]["download_success"] is True
    assert (tmp_path / "good.wav").exists()
    assert by_callsid["bad"]["download_success"] is False
    assert "404" in by_callsid["bad"]["download_info"]
    assert not (tmp_path / "bad.wav").exists()


def test_build_ground_truth_flattens_chunks_and_writes_process_csv_input(tmp_path):
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "call123.wav").write_bytes(b"fake")

    # Set up a fake per-call chunks dir + manifest, as transcribe_customer would produce
    fake_chunks_dir = tmp_path / "per_call" / "call123" / "call123_chunks"
    fake_chunks_dir.mkdir(parents=True)
    chunk_wav = fake_chunks_dir / "call123_turn001_0.00-1.00.wav"
    chunk_wav.write_bytes(b"fake wav bytes")
    fake_manifest = fake_chunks_dir / "call123_chunks_manifest.csv"
    with open(fake_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_file", "start", "end", "duration", "text"])
        writer.writeheader()
        writer.writerow(
            {"chunk_file": str(chunk_wav), "start": "0.000", "end": "1.000", "duration": "1.000", "text": "Hello there."}
        )

    class FakeResult:
        customer_chunks_dir = fake_chunks_dir
        customer_chunks_manifest_path = fake_manifest

    download_manifest_rows = [
        {
            "callsid": "call123", "region": "r0", "stt_language": "en", "stt_confidence": "0.2",
            "download_success": "True",
        }
    ]

    output_dir = tmp_path / "output"
    with patch("ingestion.ground_truth.transcribe_customer", return_value=FakeResult()):
        master_rows = build_ground_truth(download_manifest_rows, recordings_dir, output_dir)

    assert len(master_rows) == 1
    assert master_rows[0]["Filename"] == "call123_call123_turn001_0.00-1.00"
    assert master_rows[0]["Transcript"] == "Hello there."
    assert master_rows[0]["region"] == "r0"

    flattened_wav = output_dir / "benchmark_audio" / "call123_call123_turn001_0.00-1.00.wav"
    assert flattened_wav.exists()

    process_csv_path = output_dir / "process_csv_input.csv"
    with open(process_csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Filename", "Transcript"]
    assert rows[1] == ["call123_call123_turn001_0.00-1.00", "Hello there."]

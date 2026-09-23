"""Config loader for the ingestion pipeline. See config.example.json for the shape."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenSearchConfig:
    host: str
    port: int
    use_ssl: bool
    verify_certs: bool
    auth_method: str
    username: str
    password: str
    index: str
    field_callsid: str
    field_language: str
    field_confidence: str
    field_region: str
    field_recording_link: str
    field_timestamp: str


@dataclass
class AzureConfig:
    recording_link_is_direct_url: bool
    connection_string: str
    container_name: str


@dataclass
class SelectionConfig:
    start_time: str
    end_time: str
    regions: list[str]
    confidence_buckets: list[tuple[float, float]]
    samples_per_region_language_bucket: int


@dataclass
class PipelineConfig:
    opensearch: OpenSearchConfig
    azure: AzureConfig
    selection: SelectionConfig
    output_dir: Path


def load_config(path: str) -> PipelineConfig:
    data = json.loads(Path(path).read_text())
    os_cfg = data["opensearch"]
    az_cfg = data["azure"]
    sel_cfg = data["selection"]
    auth = os_cfg.get("auth", {})

    return PipelineConfig(
        opensearch=OpenSearchConfig(
            host=os_cfg["host"],
            port=os_cfg.get("port", 9200),
            use_ssl=os_cfg.get("use_ssl", True),
            verify_certs=os_cfg.get("verify_certs", True),
            auth_method=auth.get("method", "basic"),
            username=auth.get("username", ""),
            password=auth.get("password", ""),
            index=os_cfg["index"],
            field_callsid=os_cfg["fields"]["callsid"],
            field_language=os_cfg["fields"]["stt_language"],
            field_confidence=os_cfg["fields"]["stt_confidence"],
            field_region=os_cfg["fields"]["region"],
            field_recording_link=os_cfg["fields"]["recording_link"],
            field_timestamp=os_cfg["fields"]["timestamp"],
        ),
        azure=AzureConfig(
            recording_link_is_direct_url=az_cfg.get("recording_link_is_direct_url", True),
            connection_string=az_cfg.get("connection_string", ""),
            container_name=az_cfg.get("container_name", ""),
        ),
        selection=SelectionConfig(
            start_time=sel_cfg["start_time"],
            end_time=sel_cfg["end_time"],
            regions=sel_cfg["regions"],
            confidence_buckets=[tuple(b) for b in sel_cfg["confidence_buckets"]],
            samples_per_region_language_bucket=sel_cfg.get("samples_per_region_language_bucket", 5),
        ),
        output_dir=Path(data.get("output_dir", "./ingestion_output")),
    )

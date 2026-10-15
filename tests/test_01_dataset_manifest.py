"""1. Dataset manifest and checksum validation."""
from __future__ import annotations

import csv

PINNED_REVISION = "18b4b0e1359d3d02347301c55f97c3d98f5dab5c"


def test_settings_pin_dataset_revision(settings):
    dataset = settings["dataset"]
    assert dataset["huggingface_repo"] == "trucyberlab/multimodal-ICS-provenance"
    assert dataset["pinned_revision"] == PINNED_REVISION
    assert dataset["raw_data_in_repository"] is False


def test_download_manifest_present_and_checksummed(root, settings, artifact_manifest):
    relative = settings["dataset"]["manifest"]
    manifest = root / relative
    assert manifest.is_file(), f"missing {manifest}"
    from src.utils.hashing import sha256_file
    assert sha256_file(manifest) == artifact_manifest["artifacts"][relative]


def test_manifest_lists_sha256_for_every_large_file(root, settings):
    """Every LFS-backed raw file must carry a checksum so a download can be verified."""
    manifest = root / settings["dataset"]["manifest"]
    rows = list(csv.DictReader(manifest.open()))
    assert rows, "download manifest is empty"
    large = [r for r in rows if float(r.get("size_mb", 0) or 0) > 1.0]
    assert large, "no large files listed in the manifest"
    missing = [r["path"] for r in large if not (r.get("sha") or "").strip()]
    assert not missing, f"large files without a checksum: {missing}"


def test_no_raw_dataset_in_repository(root):
    """The release must not ship the raw capture or provenance files."""
    offenders = [p for p in root.rglob("*")
                 if p.is_file() and "archive" not in p.parts
                 and p.suffix in {".pcap", ".jsonl"}]
    assert not offenders, f"raw dataset files present in the release tree: {offenders[:5]}"

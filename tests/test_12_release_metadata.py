"""Public-release metadata and checksum-list checks; no detector is fitted."""
from __future__ import annotations

import json

import pytest

from src.evaluation.replay import load_artifact_manifest, load_settings
from src.utils.hashing import sha256_file


def test_internal_review_contract_is_not_in_public_package(root):
    assert not (root / "configs/paper_result_contract.json").exists()
    assert not (root / "configs/paper_result_contract.sha256").exists()


def test_release_citation_has_real_authors_and_no_placeholders(root):
    citation = (root / "CITATION.cff").read_text()
    for family in ("Ozbek", "El-Khatib", "Yang"):
        assert f"family-names: {family}" in citation
    assert "license: MIT" in citation
    assert "accepted at ICSS 2026" in citation
    for placeholder in ("AUTHOR NAME TO BE SUPPLIED", "not yet accepted", "REPOSITORY_URL"):
        assert placeholder not in citation


def test_software_and_data_terms_are_separate(root):
    license_text = (root / "LICENSE").read_text()
    notices = (root / "THIRD_PARTY_NOTICES").read_text()
    assert license_text.startswith("MIT License\n")
    assert "does not relicense the ProvICS or HAI datasets" in license_text
    assert "CC BY-NC 4.0" in notices
    assert "HAI 21.03" in notices


def test_all_saved_score_streams_are_checksummed(root, artifact_manifest):
    paths = sorted((root / "results/frozen_scores").glob("A_M*_w*.npz"))
    assert len(paths) == 16
    for path in paths:
        relative = path.relative_to(root).as_posix()
        assert artifact_manifest["artifacts"][relative] == sha256_file(path)


def test_settings_checksum_detects_edits(tmp_path, monkeypatch):
    config = tmp_path / "configs/evaluation_settings.json"
    config.parent.mkdir()
    config.write_text('{}\n')
    manifest_path = tmp_path / "data_manifest/artifact_checksums.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"artifacts": {
        "configs/evaluation_settings.json": sha256_file(config)}}))
    monkeypatch.setenv("PROVICS_ROOT", str(tmp_path))
    assert load_settings() == {}
    config.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="settings checksum mismatch"):
        load_settings()


@pytest.mark.parametrize("relative", ["../outside.csv", "/outside.csv"])
def test_manifest_rejects_paths_outside_checkout(tmp_path, monkeypatch, relative):
    config = tmp_path / "configs/evaluation_settings.json"
    config.parent.mkdir()
    config.write_text('{}\n')
    manifest_path = tmp_path / "data_manifest/artifact_checksums.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"artifacts": {relative: "0" * 64}}))
    monkeypatch.setenv("PROVICS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="artifact path must stay inside"):
        load_artifact_manifest()

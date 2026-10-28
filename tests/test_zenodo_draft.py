"""Zenodo draft metadata (no Zenodo contacted)."""
import zipfile
from pathlib import Path

import pytest
import yaml

from clara_inference.artifact import zenodo_draft as zd

MODEL_YAML = Path(__file__).parents[1] / "artifact" / "configs" / "model.yaml"
DOI = "10.5281/zenodo.123"


def _package_zip(tmp_path, placeholder_stats=False, doi=DOI):
    model = yaml.safe_load(MODEL_YAML.read_text())
    model["checkpoint"].update(doi=doi, md5="abc")
    artifact = dict(
        name="clara-2026-10-29",
        placeholder_stats=placeholder_stats,
        boundary_points=14858,
        stats={
            "danra_model1_config": {
                "description": "DANRA stats",
                "split_start": "2016-01-01 00:00:00",
                "split_end": "2019-10-22 00:00:00",
            }
        },
    )
    fp = tmp_path / "clara.zip"
    with zipfile.ZipFile(fp, "w") as zf:
        zf.writestr("artifact.yaml", yaml.safe_dump(artifact))
        zf.writestr("configs/model.yaml", yaml.safe_dump(model))
    return fp


def test_checkpoint_metadata():
    model = yaml.safe_load(MODEL_YAML.read_text())
    meta = zd.checkpoint_metadata(Path("clara_model.ckpt"), model, "1.0")
    assert meta["creators"] == zd.PAPER_AUTHORS
    assert meta["prereserve_doi"] is True
    assert "--backbone_model graph_unet2" in meta["description"]


def test_package_requires_the_checkpoint(tmp_path):
    meta = zd.package_metadata(_package_zip(tmp_path), "1.0")
    assert meta["creators"] == zd.PAPER_AUTHORS + zd.PACKAGE_AUTHORS
    requires = [
        r for r in meta["related_identifiers"] if r["relation"] == "requires"
    ]
    assert requires[0]["identifier"] == DOI
    assert "does not contain the model weights" in meta["description"]
    assert "2016-01-01" in meta["description"]


def test_package_refuses_placeholder_stats(tmp_path):
    with pytest.raises(ValueError, match="placeholder"):
        zd.package_metadata(
            _package_zip(tmp_path, placeholder_stats=True), "1.0"
        )


def test_package_needs_the_checkpoint_doi(tmp_path):
    with pytest.raises(ValueError, match="DOI"):
        zd.package_metadata(_package_zip(tmp_path, doi=None), "1.0")

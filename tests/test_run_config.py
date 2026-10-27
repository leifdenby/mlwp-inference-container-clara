"""The configuration of the entrypoint (anna_inference.run)."""
import datetime
import re
from pathlib import Path

import pytest

from anna_inference import envargs, run

README = Path(__file__).parents[1] / "README.md"


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "model.yaml").write_text("{}\n")
    return tmp_path


def test_readme_option_table_matches_the_parser():
    """The README's table lists every option, with its env var name."""
    rows = re.findall(
        r"^\| `(--[a-z0-9-]+)[^`]*` \| `(ANNA_[A-Z0-9_]+)` \|",
        README.read_text(),
        flags=re.MULTILINE,
    )
    parser = run.build_parser()
    expected = {
        option: envargs.env_var_name(run.ENV_PREFIX, option)
        for option in (opt for _, opt in envargs._options(parser))
    }
    assert dict(rows) == expected


def test_config_from_env(artifact, tmp_path):
    environ = {
        "ANNA_ANALYSIS_TIME": "2026-10-01T06:00+02:00",
        "ANNA_FORECAST_DURATION": "PT30H",
        "ANNA_ARTIFACT_PATH": str(artifact),
        "ANNA_WORKDIR": str(tmp_path / "work"),
    }
    _, args, sources = run.parse_config([], environ)
    assert args.analysis_time == datetime.datetime(2026, 10, 1, 4)
    assert args.forecast_duration == datetime.timedelta(hours=30)
    assert args.dini_root == (
        "s3://harmonie-zarr/dini/control/2026-10-01T040000Z/"
    )
    assert sources["forecast_duration"] == "env"
    assert sources["boundary_source"] == "default"


@pytest.mark.parametrize("duration", ["PT6H", "PT13H", "P1M", "18h"])
def test_invalid_forecast_duration(artifact, duration):
    argv = ["--analysis-time", "2026-10-01T00:00"]
    environ = {
        "ANNA_FORECAST_DURATION": duration,
        "ANNA_ARTIFACT_PATH": str(artifact),
    }
    with pytest.raises(SystemExit) as ex:
        run.parse_config(argv, environ)
    assert ex.value.code == 2


def test_ifs_needs_a_boundary_path(artifact):
    argv = [
        "--analysis-time", "2026-10-01T00:00",
        "--boundary-source", "ifs",
        "--artifact-path", str(artifact),
    ]  # fmt: skip
    with pytest.raises(SystemExit) as ex:
        run.parse_config(argv, {})
    assert ex.value.code == 2


def test_missing_artifact(tmp_path):
    argv = ["--analysis-time", "2026-10-01T00:00"]
    with pytest.raises(SystemExit) as ex:
        run.parse_config(argv, {"ANNA_ARTIFACT_PATH": str(tmp_path)})
    assert ex.value.code == 2


AMBIENT = {
    "AWS_PROFILE": "ambient",
    "AWS_ACCESS_KEY_ID": "key",
    "AWS_SECRET_ACCESS_KEY": "secret",
}


def _forecast(artifact, monkeypatch, environ):
    for name, value in AMBIENT.items():
        monkeypatch.setenv(name, value)
    argv = ["--analysis-time", "2026-10-01T00:00"]
    environ = {"ANNA_ARTIFACT_PATH": str(artifact), **environ}
    _, args, _ = run.parse_config(argv, environ)
    return run.Forecast(args)


def test_step_env_per_side(artifact, monkeypatch):
    forecast = _forecast(
        artifact,
        monkeypatch,
        {
            "ANNA_INPUT_S3_ENDPOINT_URL": "https://in.example",
            "ANNA_OUTPUT_AWS_PROFILE": "writer",
            "ANNA_OUTPUT_S3_PREFIX": "s3://bucket/",
        },
    )
    env_in = forecast.step_env("input")
    assert env_in["AWS_ENDPOINT_URL_S3"] == "https://in.example"
    assert env_in["AWS_ENDPOINT_URL"] == "https://in.example"
    # no input profile: the ambient credentials stay
    assert {k: env_in[k] for k in AMBIENT} == AMBIENT

    env_out = forecast.step_env("output")
    assert "AWS_ENDPOINT_URL_S3" not in env_out
    assert env_out["AWS_PROFILE"] == "writer"
    # the profile wins over credentials in the environment
    assert "AWS_ACCESS_KEY_ID" not in env_out
    assert "AWS_SECRET_ACCESS_KEY" not in env_out


def test_step_env_unchanged_without_options(artifact, monkeypatch):
    forecast = _forecast(artifact, monkeypatch, {})
    for side in ["input", "output"]:
        assert forecast.step_env(side) == forecast.env


@pytest.mark.parametrize(
    "name", ["ANNA_OUTPUT_S3_ENDPOINT_URL", "ANNA_OUTPUT_AWS_PROFILE"]
)
def test_output_options_need_a_prefix(artifact, name):
    argv = ["--analysis-time", "2026-10-01T00:00"]
    environ = {"ANNA_ARTIFACT_PATH": str(artifact), name: "x"}
    with pytest.raises(SystemExit) as ex:
        run.parse_config(argv, environ)
    assert ex.value.code == 2

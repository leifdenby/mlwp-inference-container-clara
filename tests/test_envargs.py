"""The generic CLI <-> {PREFIX}_* environment variable helper."""
import argparse

import pytest

from anna_inference import envargs

PREFIX = "MODEL"


def _parser():
    parser = argparse.ArgumentParser(prog="model")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--mode", choices=["a", "b"], default="a")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction)
    parser.add_argument("--levels", nargs="+", type=int, default=[1000])
    return parser


def _parse(argv, environ):
    return envargs.parse_args(_parser(), PREFIX, argv, environ)


def test_env_var_name():
    name = envargs.env_var_name("ANNA", "--forecast-duration")
    assert name == "ANNA_FORECAST_DURATION"


def test_env_sets_default_and_cli_wins():
    environ = {"MODEL_INPUT_PATH": "env.zarr", "MODEL_N_STEPS": "8"}
    args, sources = _parse([], environ)
    assert (args.input_path, args.n_steps) == ("env.zarr", 8)
    assert sources["input_path"] == sources["n_steps"] == "env"
    assert sources["mode"] == "default"

    args, sources = _parse(["--n-steps", "2"], environ)
    assert (args.input_path, args.n_steps) == ("env.zarr", 2)
    assert sources["n_steps"] == "cli"


def test_required_from_env_or_missing():
    args, _ = _parse([], {"MODEL_INPUT_PATH": "x"})
    assert args.input_path == "x"
    with pytest.raises(SystemExit) as ex:
        _parse([], {})
    assert ex.value.code == 2


def test_empty_variable_is_unset():
    args, sources = _parse(["--input-path", "x"], {"MODEL_N_STEPS": ""})
    assert args.n_steps == 4 and sources["n_steps"] == "default"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("1", True),
        ("YES", True),
        ("false", False),
        ("0", False),
    ],
)
def test_flags(value, expected):
    environ = {"MODEL_INPUT_PATH": "x", "MODEL_DRY_RUN": value}
    args, _ = _parse([], environ)
    assert args.dry_run is expected

    environ = {"MODEL_INPUT_PATH": "x", "MODEL_CACHE": value}
    args, _ = _parse([], environ)
    assert args.cache is expected


def test_invalid_flag(capsys):
    with pytest.raises(SystemExit):
        _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_DRY_RUN": "maybe"})
    assert "MODEL_DRY_RUN='maybe'" in capsys.readouterr().err


def test_nargs_from_space_separated_value():
    args, _ = _parse([], {"MODEL_INPUT_PATH": "x", "MODEL_LEVELS": "850 500"})
    assert args.levels == [850, 500]


@pytest.mark.parametrize(
    "name, value", [("MODEL_N_STEPS", "many"), ("MODEL_MODE", "c")]
)
def test_type_and_choice_errors_name_the_variable(capsys, name, value):
    with pytest.raises(SystemExit) as ex:
        _parse([], {"MODEL_INPUT_PATH": "x", name: value})
    assert ex.value.code == 2
    assert f"{name}={value!r}" in capsys.readouterr().err


def test_unknown_prefixed_variable(capsys):
    environ = {"MODEL_INPUT_PATH": "x", "MODEL_N_STEP": "8", "OTHER_X": "1"}
    with pytest.raises(SystemExit) as ex:
        _parse([], environ)
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "MODEL_N_STEP (did you mean MODEL_N_STEPS?)" in err
    assert "OTHER_X" not in err


def test_help_and_config_lines(capsys):
    parser = _parser()
    envargs.add_env_help(parser, PREFIX)
    assert "[env: MODEL_N_STEPS]" in parser.format_help()

    args, sources = envargs.parse_args(
        parser, PREFIX, ["--mode", "b"], {"MODEL_INPUT_PATH": "x"}
    )
    lines = envargs.config_lines(parser, args, sources, PREFIX)
    assert any(
        line.split() == ["MODEL_INPUT_PATH", "=", "x", "(env)"]
        for line in lines
    )
    assert any(
        line.split() == ["MODEL_MODE", "=", "b", "(cli)"] for line in lines
    )


def test_positional_arguments_unsupported():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    with pytest.raises(ValueError):
        envargs.parse_args(parser, PREFIX, ["x"], {})

"""
Run a CLARA ensemble forecast (CRPS-LAM Rapid Atmospheric ensembles: the
CRPS-LAM model of arXiv:2510.09484, see the artifact's configs/model.yaml)
from a DINI control forecast (interior) and an IFS forecast (boundary).

This is the container's entrypoint (`python -m clara_inference`), where the
inference artifact is in ./inference_artifact. It can also be run outside
the container in the uv environment of this repository, with
`--artifact-path artifact/build/<artifact-name>`.

Steps, each run as a subprocess:
  1. regrid DINI to the DANRA grid every 3 h (clara_inference.regrid_dini)
  2. create the inference datastores (interior, and the IFS boundary in
     training point order) and the neural-lam config, and link the graph
                                   (clara_inference.create_inference_dataset)
  3. run the ensemble forecast            (neural_lam.train_model --eval)
  4. convert the prediction to DANRA-like single/pressure level zarr
     datasets with an ensemble_member dimension
                                          (clara_inference.convert_output)
  5. optionally, upload the outputs to S3 (clara_inference.upload_outputs)

Outputs: {workdir}/outputs/{single_levels,pressure_levels}.zarr

Exit codes: 0 success, 2 configuration error, 1 a failed step.
"""
import argparse
import datetime
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import dotenv
import isodate
import yaml
from loguru import logger

import clara_inference
from clara_inference import envargs

ENV_PREFIX = "CLARA"
DEFAULT_DINI_ROOT = "s3://harmonie-zarr/dini/control/{analysis}/"
DEFAULT_ENSEMBLE_SIZE = 5
AWS_ENV_CREDENTIALS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
]

# as in create_inference_dataset.py (not imported, to keep --help fast): the
# interior time step, and the shortest forecast neural-lam can load the model
# with
INTERIOR_STEP = datetime.timedelta(hours=3)
MIN_FORECAST_DURATION = datetime.timedelta(hours=12)


class StepFailed(Exception):
    pass


def parse_analysis_time(s):
    """ISO8601, UTC if no timezone; returned as naive UTC."""
    try:
        t = isodate.parse_datetime(s)
    except (ValueError, isodate.ISO8601Error) as ex:
        raise argparse.ArgumentTypeError(str(ex))
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def parse_forecast_duration(s):
    """ISO8601 duration, a multiple of 3h and at least 12h."""
    try:
        duration = isodate.parse_duration(s)
    except (ValueError, isodate.ISO8601Error) as ex:
        raise argparse.ArgumentTypeError(str(ex))
    if (
        not isinstance(duration, datetime.timedelta)
        or duration < MIN_FORECAST_DURATION
        or duration % INTERIOR_STEP
    ):
        raise argparse.ArgumentTypeError(
            f"must be a multiple of 3h and at least 12h, got {s}"
        )
    return duration


def parse_ensemble_size(s):
    try:
        n = int(s)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(str(ex))
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {s}")
    return n


def build_parser():
    parser = argparse.ArgumentParser(
        prog="clara-inference",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis-time",
        required=True,
        type=parse_analysis_time,
        help="DINI analysis time, ISO8601, e.g. 2026-10-01T06:00Z (UTC if "
        "no timezone)",
    )
    parser.add_argument(
        "--forecast-duration",
        type=parse_forecast_duration,
        default="PT18H",
        help="ISO8601 duration, a multiple of 3h, from PT12H (neural-lam "
        "needs that to load the model) to the DINI forecast length - 6h "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--ensemble-size",
        type=parse_ensemble_size,
        default=DEFAULT_ENSEMBLE_SIZE,
        help="number of ensemble members (default: %(default)s)",
    )
    parser.add_argument(
        "--ifs-boundary-path",
        required=True,
        help="IFS forecast zarr (local or S3) following the contract in "
        "clara_inference/ifs_boundary.py; the latest 00/12 UTC cycle at or "
        "before analysis time - 3h is used",
    )
    parser.add_argument(
        "--dini-root",
        help=f"DINI forecast zarr root (default: {DEFAULT_DINI_ROOT})",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("inference_workdir"),
        help="working directory, with the outputs in outputs/ "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("inference_artifact"),
        help="the inference artifact, with the checkpoint (default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="skip the regridding and datastore steps already completed in "
        "the workdir (marked by a .completed file), e.g. to rerun only the "
        "forecast. Only use with the same analysis time, forecast duration "
        "and IFS boundary as the completed run",
    )
    parser.add_argument(
        "--output-s3-prefix",
        help="also upload the outputs to {prefix}/{analysis time}/ (e.g. "
        "s3://clara-forecasts/2026-10-01T060000Z/), replacing an earlier "
        "upload for that analysis time",
    )
    # S3 endpoint and credentials per side (input: DINI and IFS; output: the
    # upload). Unset, the usual AWS configuration applies (AWS_* variables,
    # ~/.aws)
    for side, what in [
        ("input", "reading DINI and IFS"),
        ("output", "the upload"),
    ]:
        parser.add_argument(
            f"--{side}-s3-endpoint-url",
            help=f"S3 endpoint URL for {what} (default: from the AWS "
            "configuration)",
        )
        parser.add_argument(
            f"--{side}-aws-profile",
            help=f"AWS profile (in ~/.aws) for {what}; overrides AWS_* "
            "credentials in the environment (default: from the AWS "
            "configuration)",
        )
    parser.add_argument(
        "--debugger",
        choices=["ipdb"],
        help="debug the python steps on exceptions",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="loguru level (default: %(default)s)",
    )
    envargs.add_env_help(parser, ENV_PREFIX)
    return parser


def parse_config(argv=None, environ=None):
    """The parsed and checked configuration, and each value's source."""
    parser = build_parser()
    args, sources = envargs.parse_args(parser, ENV_PREFIX, argv, environ)
    if not args.output_s3_prefix and (
        args.output_s3_endpoint_url or args.output_aws_profile
    ):
        parser.error(
            "--output-s3-endpoint-url/--output-aws-profile need "
            "--output-s3-prefix (nothing is uploaded without it)"
        )
    if args.dini_root is None:
        compact = f"{args.analysis_time:%Y-%m-%dT%H%M%SZ}"
        args.dini_root = DEFAULT_DINI_ROOT.format(analysis=compact)
    args.artifact_path = args.artifact_path.resolve()
    if not (args.artifact_path / "configs" / "model.yaml").is_file():
        parser.error(
            f"{args.artifact_path} isn't an inference artifact "
            "(no configs/model.yaml)"
        )
    args.workdir = args.workdir.resolve()
    return parser, args, sources


class Forecast:
    """The pipeline's steps for one configuration."""

    def __init__(self, args):
        self.args = args
        self.workdir = args.workdir
        self.artifact = args.artifact_path
        self.analysis_iso = f"{args.analysis_time:%Y-%m-%dT%H:%M}"
        self.duration_iso = isodate.duration_isoformat(args.forecast_duration)
        self.ar_steps = int(args.forecast_duration / INTERIOR_STEP) - 1
        self.model = yaml.safe_load(
            (self.artifact / "configs" / "model.yaml").read_text()
        )
        self.interior_name = Path(self.model["interior_datastore"]).stem
        self.danra_grid = (
            self.artifact / "grids" / f"{self.interior_name}.grid.zarr"
        )
        self.env = dict(os.environ)
        if args.debugger:
            # read by the clara_inference steps
            self.env["CLARA_DEBUGGER"] = args.debugger

    def step_env(self, side):
        """
        The environment of a step reading ("input") or writing ("output")
        S3, with that side's endpoint and profile, which botocore reads
        from AWS_ENDPOINT_URL_S3/AWS_ENDPOINT_URL and AWS_PROFILE.
        """
        env = dict(self.env)
        endpoint_url = getattr(self.args, f"{side}_s3_endpoint_url")
        profile = getattr(self.args, f"{side}_aws_profile")
        if endpoint_url:
            env["AWS_ENDPOINT_URL_S3"] = endpoint_url
            env["AWS_ENDPOINT_URL"] = endpoint_url
        if profile:
            env["AWS_PROFILE"] = profile
            # botocore prefers credentials in the environment over a profile
            for name in AWS_ENV_CREDENTIALS:
                env.pop(name, None)
        return env

    def run_step(self, name, cmd, cwd=None, env=None):
        logger.info(f"{name}: {shlex.join(map(str, cmd))}")
        if env is not None:
            s3 = {
                k: env[k]
                for k in ["AWS_ENDPOINT_URL_S3", "AWS_PROFILE"]
                if k in env and env[k] != self.env.get(k)
            }
            if s3:
                logger.info(f"{name}: with {s3}")
        result = subprocess.run(
            [str(c) for c in cmd], cwd=cwd, env=env or self.env
        )
        if result.returncode != 0:
            raise StepFailed(
                f"{name} failed with exit code {result.returncode}"
            )

    def completed(self, path, what):
        """Whether to skip a step, with --skip-completed."""
        if self.args.skip_completed and (path / ".completed").exists():
            logger.info(f"Skipping {what}: already completed in {path}")
            return True
        return False

    def mark_completed(self, path):
        (path / ".completed").write_text(
            f"{self.analysis_iso} {self.duration_iso} "
            f"{self.args.ifs_boundary_path}\n"
        )

    def checkpoint(self):
        checkpoint = self.artifact / self.model["checkpoint"]["artifact_path"]
        if not checkpoint.is_file():
            raise StepFailed(
                f"model checkpoint {checkpoint} not found, download it with "
                "`python -m clara_inference.fetch_checkpoint "
                f"{self.artifact}`"
            )
        return checkpoint

    def regrid(self):
        inputs = self.workdir / "inputs"
        if self.completed(inputs, "the regridding"):
            return
        (inputs / ".completed").unlink(missing_ok=True)
        self.run_step(
            "1. regrid DINI",
            [
                sys.executable,
                "-m",
                "clara_inference.regrid_dini",
                "regrid",
                "--dini-root",
                self.args.dini_root,
                "--danra-grid",
                self.danra_grid,
                "--forecast-duration",
                self.duration_iso,
                "--output",
                inputs,
            ],
            env=self.step_env("input"),
        )
        self.mark_completed(inputs)

    def create_datastores(self):
        datastores = self.workdir / "datastores"
        if self.completed(datastores, "the inference datastores"):
            return
        (datastores / ".completed").unlink(missing_ok=True)
        self.run_step(
            "2. inference datastores",
            [
                sys.executable,
                "-m",
                "clara_inference.create_inference_dataset",
                "--artifact",
                self.artifact,
                "--interior-dir",
                self.workdir / "inputs",
                "--ifs-boundary",
                self.args.ifs_boundary_path,
                "--analysis-time",
                self.analysis_iso,
                "--forecast-duration",
                self.duration_iso,
                "--workdir",
                datastores,
            ],
            # reads the IFS boundary
            env=self.step_env("input"),
        )
        self.mark_completed(datastores)

    @property
    def nl_config(self):
        return self.workdir / "datastores" / "config.yaml"

    def forecast(self, checkpoint):
        """
        - strict checkpoint loading (NEURAL_LAM_STRICT_LOAD=1): neural-lam
          would otherwise keep randomly initialised weights for missing or
          mismatched parameters
        - no test metrics (NEURAL_LAM_SKIP_TEST_METRICS=1): the "targets"
          are the DINI forecast, and the metric plots fail for one sample
        - wandb offline (neural-lam always creates a wandb logger); it
          writes its run dir relative to the CWD
        - empty --eval_init_times: no restriction to 00/12 UTC forecasts
        - neural-lam's main() only logs exceptions (exit code 0), so check
          the output
        """
        prediction = self.workdir / "outputs" / "prediction.zarr"
        if prediction.exists():
            shutil.rmtree(prediction)
        (self.workdir / "outputs").mkdir(parents=True, exist_ok=True)
        wandb_dir = self.workdir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **self.env,
            "WANDB_MODE": "offline",
            "WANDB_DIR": wandb_dir,
            "NEURAL_LAM_STRICT_LOAD": "1",
            # no metrics/plots against the (DINI) "targets" at the end
            "NEURAL_LAM_SKIP_TEST_METRICS": "1",
        }
        self.run_step(
            "3. forecast",
            [
                sys.executable,
                "-m",
                "neural_lam.train_model",
                "--config_path",
                self.nl_config,
                *map(str, self.model["train_model_args"]),
                "--graph_name",
                self.model["graph"]["name"],
                "--ensemble_size",
                self.args.ensemble_size,
                "--eval",
                "test",
                "--ar_steps_eval",
                self.ar_steps,
                "--val_steps_to_log",
                *range(1, self.ar_steps + 1),
                "--eval_init_times",
                "--n_example_pred",
                0,
                "--batch_size",
                1,
                "--num_workers",
                0,
                "--load",
                checkpoint,
                "--save_eval_to_zarr_path",
                prediction,
            ],
            cwd=self.workdir,
            env={k: str(v) for k, v in env.items()},
        )
        if not prediction.is_dir():
            raise StepFailed(
                f"the forecast didn't produce {prediction}, see the "
                "neural-lam log above"
            )
        return prediction

    def convert(self, prediction):
        self.run_step(
            "4. convert outputs",
            [
                sys.executable,
                "-m",
                "clara_inference.convert_output",
                "--prediction",
                prediction,
                "--danra-grid",
                self.danra_grid,
                "--interior-config",
                self.artifact / "configs" / self.model["interior_datastore"],
                "--analysis-time",
                self.analysis_iso,
                "--output-dir",
                self.workdir / "outputs",
            ],
        )

    def upload(self):
        self.run_step(
            "5. upload outputs",
            [
                sys.executable,
                "-m",
                "clara_inference.upload_outputs",
                "--outputs",
                self.workdir / "outputs",
                "--analysis-time",
                self.analysis_iso,
                "--prefix",
                self.args.output_s3_prefix,
            ],
            env=self.step_env("output"),
        )

    def run(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        artifact_yaml = self.artifact / "artifact.yaml"
        if artifact_yaml.exists() and "placeholder_stats: true" in (
            artifact_yaml.read_text()
        ):
            logger.warning(
                "the inference artifact uses PLACEHOLDER normalisation "
                "statistics, not the training statistics: for testing only"
            )
        log_device()
        checkpoint = self.checkpoint()
        self.regrid()
        self.create_datastores()
        prediction = self.forecast(checkpoint)
        self.convert(prediction)
        if self.args.output_s3_prefix:
            self.upload()
        logger.info(
            f"CLARA forecast done: {self.workdir}/outputs/"
            "{single_levels,pressure_levels}.zarr"
        )


def log_device():
    import torch

    logger.info(
        f"torch {torch.__version__}, cuda available: "
        f"{torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        logger.info(f"device: {torch.cuda.get_device_name(0)}")


def main(argv=None):
    logger.info(f"Starting clara-inference {clara_inference.__version__}")
    # a local .env file (e.g. with S3 credentials, or CLARA_* variables);
    # variables already set win
    if Path(".env").is_file():
        logger.info("Loading environment variables from .env")
        dotenv.load_dotenv(".env", override=False)
    parser, args, sources = parse_config(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    logger.info("Effective configuration (defaults included):")
    for line in envargs.config_lines(parser, args, sources, ENV_PREFIX):
        logger.info(f"  {line}")
    forecast = Forecast(args)
    logger.info(
        f"{forecast.duration_iso} forecast from {forecast.analysis_iso} "
        f"({forecast.ar_steps} steps, {args.ensemble_size} members)"
    )
    try:
        forecast.run()
    except StepFailed as ex:
        logger.error(str(ex))
        sys.exit(1)


if __name__ == "__main__":
    main()

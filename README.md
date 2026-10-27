# mlwp-inference-container-clara

> **Status: pre-release.** The pipeline runs end to end. The inference
> artifact with the full training statistics is still to be built and
> published; see [PLAN.md](PLAN.md) for what is done and what remains.

This repository builds the inference container images for **CLARA**,
**C**RPS-**LA**M **R**apid **A**tmospheric ensembles. CLARA runs **CRPS-LAM**
operationally. "Rapid" refers to the model's main strength: each ensemble
member is generated in a single forward pass, about 39× faster than diffusion
baselines. CRPS-LAM is the probabilistic machine-learning limited-area model
of Larsson et al. (2025), ["CRPS-LAM: Probabilistic Regional Weather
Forecasting with Continuous Ranked Probability
Score"](https://arxiv.org/abs/2510.09484).

- **Interior initial states:** **DINI**, regridded to the DANRA grid.
- **Boundary:** **IFS** forecasts.
- **Output:** an **ensemble forecast**.

**Model architecture:** a hybrid GNN/UNet backbone, neural-lam `crps` with
backbone `graph_unet2` and graph `feature_map_graph_198x148`. A noise vector
is injected at every step, so each ensemble member is drawn in a single
forward pass. It was trained with a CRPS loss.

**Training data:**
- **Interior:** DANRA, 2.5 km and 3-hourly.
- **Boundary:** first ERA5, then fine-tuned on IFS forecasts, on a 400 km
  margin around the DANRA domain.

Everything needed for that is in one Python project
([pyproject.toml](pyproject.toml), with the exact environment in `uv.lock`):
- **Inference pipeline:** `src/clara_inference/`, run by
  `python -m clara_inference`, which is the image's entrypoint.
- **Image build:** the [Containerfile](Containerfile) and
  [scripts/build_image.sh](scripts/build_image.sh).
- **Artifact tools:** the tools that build the model's *inference artifact*
  (configs, training statistics, grids, graph, checkpoint; see below), in
  `src/clara_inference/artifact/`.

The design follows
[`dmidk/mlwp-inference-container-anna`](https://github.com/dmidk/mlwp-inference-container-anna)
(ANNA, the deterministic DANRA model). Much of the code is shared with it,
including the option/environment convention, checkpoint fetching, DINI
regridding and the S3 upload.

## Model

The model is described in
[artifact/configs/model.yaml](artifact/configs/model.yaml): the checkpoint,
the graph, the neural-lam arguments and the training neural-lam config.

**Paper:** E. Larsson, J. Oskarsson, T. Landelius, F. Lindsten,
[arXiv:2510.09484](https://arxiv.org/abs/2510.09484).

**Checkpoint:** `CRPS_IFS_ALL_FINETUNE_AR12-train-crps-4x330-05_21_13-3613`.
- wandb project `real_prob_lam`, run `CRPS_IFS_ALL_FINETUNE_AR12`, epoch 92.
- Fine-tuned with 12-step rollouts and an IFS boundary, starting from
  `CRPS_IFS_ALL_FINETUNE-train-crps-4x330-05_20_08-5709`.
- About 251 M parameters, trained in `bf16-mixed`.

**Key settings.** These are all the checkpoint arguments that affect the
model and differ from `train_model`'s defaults:

| Setting | Value |
|---|---|
| Model / backbone | `--model crps --backbone_model graph_unet2` |
| Widths | `--hidden_dim 330 --hidden_dim_grid 270 --hidden_dim_mesh_nodes 270 --hidden_dim_edge 32` |
| UNet | `--channel_mult 1 1 1 1 1 1 --attn_res 10 18 36 70` |
| Noise | `--noise_dim 32 --noise_embedding linear` |
| Prediction | `--pred_residual`, `--precision bf16-mixed` |
| Ensemble | `--ensemble_size` is configurable (default 5) |

**Training code:** neural-lam from
[`leifdenby/real-prob-lam`](https://github.com/leifdenby/real-prob-lam)
(`neural-lam-dev-building-ml-lams/`), a fork of `joeloskarsson/neural-lam-dev`.

## Software

Pinned in [pyproject.toml](pyproject.toml):

- **neural-lam:** `leifdenby/real-prob-lam@47cc2fe` (subdirectory
  `neural-lam-dev-building-ml-lams`, branch `prob-model-inference`). This
  repository is private, see [Building the image](#building-the-image). The
  branch adds:
  - the bf16 and single-GPU inference fixes;
  - `NEURAL_LAM_STRICT_LOAD=1`, which makes loading a checkpoint fail instead
    of silently keeping randomly initialised weights for missing or
    mismatched parameters. The pipeline always sets it.
- **mllam-data-prep:** `Erik-Wikingsson/mllam-data-prep-real@0589539`
  (`rect-boundary`, with `latlon-domain-crop`), as used for training.
- **torch 2.12** (CUDA 13.0 index) in the uv environment. The image uses its
  base image's torch.
- **zarr 2**, as in training.

## Domain

![The CLARA domain: DANRA interior, IFS boundary points and the DINI domain](docs/clara_domain.png)

- **Interior:** the full DANRA grid, 2.5 km Lambert conformal with 589 × 789
  points. Initial states are regridded to it from DINI.
- **Boundary:** 14,858 points of the 0.25° grid within 400 km of the
  interior. Unlike ANNA's boundary ring, the points *inside* the interior
  (orange) are included too. All of them come from IFS, so, unlike ANNA's DINI
  boundary, there is no coverage gap.

The figure is made with `dev-utils/plot_domain.py`.

## Configs

The trained model was set up with these configs, in
[artifact/configs/](artifact/configs/) (with the training cluster's paths
replaced by the public sources, or by placeholders set at inference):

| File | What |
|---|---|
| `danra_model_config_ifs_all.yaml` | the neural-lam config the model was trained with: state feature weights, clamping of `r` to [0, 1] |
| `danra_model1_config_ifs_finetune.yaml` | interior (DANRA) datastore, PT3H. 55 state features: pres_seasurface, t2m, u10m, v10m, pres0m, lwavr0m, swavr0m, and z/t/r/u/v/tw at 100–1000 hPa. Statics lsm, orography; forcings toa_radiation and time of day/year |
| `overlapping_ifs_all2_400km_model1_config.yaml` | boundary (IFS forecast) datastore. Forecast layout `analysis_time × elapsed_forecast_duration` with lead times every 6 h, 400 km margin (14,858 points of the 0.25° grid), 58 forcing features with ERA5 names, ERA5 statics |
| `danra_model1_config.yaml` | the interior statistics datastore (DANRA, 2016-01-01 → 2019-10-22) |
| `overlapping_era_400km_model1_config.yaml` | the boundary statistics datastore (ERA5, same period): the IFS boundary is normalised with ERA5 statistics (`overload_stats_path`) |

## Inference artifact

An **inference artifact** is a directory, shipped as a zip, that holds
everything needed to run a trained model's forecasts apart from the input data
and the software. See ANNA's README for the concept. The layout is described
in [layout.py](src/clara_inference/artifact/layout.py), and the README shipped
in the zip is [artifact/package-README.md](artifact/package-README.md).

| Path | What |
|---|---|
| `configs/` | the training configs above and `model.yaml` |
| `configs/<stats datastore>.zarr` | next to the two statistics datastore configs: the training statistics, which neural-lam reads for `overload_stats_path` |
| `grids/` | the DANRA grid and statics (target grid for regridding DINI), and the boundary grid with the ERA5 statics |
| `graphs/feature_map_graph_198x148/` | the graph the model was trained with (not rebuilt at run time) |
| `artifact.yaml`, `README.md` | provenance and description |
| `clara_model.ckpt` | the sanitized checkpoint (1.0 GB): weights, plus `args` and `config` as plain dicts. No optimizer state or pickled training datastores, so it loads with `weights_only=True` |

### Training statistics

**The normalisation statistics are not in the checkpoint.** neural-lam
registers them as non-persistent buffers, and the training statistics only
existed on the training cluster.
[compute_stats](src/clara_inference/artifact/compute_stats.py) recomputes
them from the public data over the training period:
- **Interior:** DANRA v0.5.0 from the ECMWF object store.
- **Boundary:** WeatherBench2 ERA5.

It computes exactly what mllam-data-prep computes, block by block:
- mean and std of the values;
- mean of the **first** time differences;
- std of the **second** time differences. This is what mllam-data-prep's
  `calc_stats` really computes for `diff_std`.

On real data, the result matches mllam-data-prep to within 2.5e-8 of each
feature's spread.

Statistics from only a few days are **unusable**. The day-of-year forcings
then have a std close to 0, which puts their standardised values at about
−1600σ, and the forecast diverges.

### Building the artifact

```bash
# the training statistics (reads ~1 TB per datastore; hours)
uv run python -m clara_inference.artifact.compute_stats \
    --configs artifact/configs --output-dir artifact/build/stats
# the artifact (DANRA grid from the object store, ERA5 statics from WB2)
uv run python -m clara_inference.artifact.assemble \
    --output artifact/build/clara-local \
    --checkpoint <training checkpoint>/last.ckpt \
    --graph <...>/feature_map_graph_198x148.zip \
    --stats-dir artifact/build/stats --zip artifact/build/clara-local.zip
```

`assemble` derives the boundary grid from the graph's boundary nodes, and
checks the artifact before finishing (`layout.check_artifact`).

## Building the image

```bash
CONTAINER_APP=docker scripts/build_image.sh
```

The image is tagged `mlwp-inference-container-clara:latest` (`IMAGE_NAME`).

- **Token.** neural-lam is installed from the private `leifdenby/real-prob-lam`.
  The build needs a GitHub token: `GITHUB_TOKEN`, or the one from
  `gh auth login`. It is passed as a BuildKit secret, never as a build
  argument, so it doesn't end up in the image.
- **torch.** The install uses the base image's torch, not the cu130 index of
  the uv environment. The git pins of `pyproject.toml` are passed to
  `uv pip install --no-sources` by
  [scripts/container_pins.py](scripts/container_pins.py).
- **`ARTIFACT_SOURCE=local`** (default) uses an assembled artifact directory,
  `LOCAL_ARTIFACT_DIR` (default `artifact/build/clara-local`).
- **`ARTIFACT_SOURCE=release`** downloads the package zip, a release asset of
  this repository: `PACKAGE_URL` is a GitHub API asset URL and is checked
  against `PACKAGE_MD5`.
- **Checkpoint.** It is downloaded with the token, md5-checked, from
  `model.yaml`'s url, unless the artifact already has it.

**CI** ([container-image.yml](.github/workflows/container-image.yml)) builds
the x86_64 image on every push and pull request. `v*` tags also publish it as
`ghcr.io/leifdenby/mlwp-inference-container-clara:<tag>` and `:latest`. It
needs:
- the repository secret `REAL_PROB_LAM_TOKEN`: a fine-grained token with read
  access to the contents of `leifdenby/real-prob-lam` and of this repository;
- the variables `PACKAGE_URL` and `PACKAGE_MD5`.

Without them, the build is skipped.

## Running a forecast

`python -m clara_inference` runs the whole pipeline. It is also installed as
`clara-inference`, which is the image's entrypoint. Each step runs as a
subprocess:

1. **`clara_inference.regrid_dini`:** regrids DINI to the DANRA grid at
   3-hourly steps.
2. **`clara_inference.create_inference_dataset`:** builds the inference
   datastores and the neural-lam config.
   - The interior is normalised with the DANRA training statistics.
   - The boundary uses the selected IFS cycle on the boundary grid, normalised
     with the ERA5 training statistics.
   - It checks the boundary points against the graph's boundary nodes, and
     that the statistics' features match the datastores' features in order.
   - The artifact's graph is linked in.
3. **`neural_lam.train_model --eval test`:** runs the ensemble forecast with
   strict checkpoint loading.
4. **`clara_inference.convert_output`:** writes DANRA-like output datasets
   with an `ensemble_member` dimension.
5. **`clara_inference.upload_outputs`:** optionally uploads to S3.

A 5-member forecast of 18 h runs the model in about 2 minutes on a DGX Spark
(GB10).

```bash
uv run python -m clara_inference --artifact-path artifact/build/clara-local \
    --analysis-time 2026-10-26T06:00Z --forecast-duration PT18H \
    --ifs-boundary-path dev-data/ifs/2026-10-26T000000Z/ifs.zarr
# in the container
CONTAINER_APP=docker CLARA_IFS_BOUNDARY_PATH=dev-data/ifs/2026-10-26T000000Z/ifs.zarr \
    scripts/run_inference_container.sh 2026-10-26T06:00Z PT18H
```

**S3 access.** S3 inputs are read with the usual AWS configuration. Without
any AWS credentials they are read anonymously, which works for DINI's public
`harmonie-zarr` bucket.

### Configuration

Every command-line option `--some-option` can also be set with the
environment variable `CLARA_SOME_OPTION`. The command line wins over the
variable, which wins over the default. This is the same convention as ANNA
(`envargs.py`).

| Option | Environment variable | Default | |
|---|---|---|---|
| `--analysis-time` | `CLARA_ANALYSIS_TIME` | (required) | DINI analysis time, ISO8601, e.g. `2026-10-01T06:00Z` |
| `--forecast-duration` | `CLARA_FORECAST_DURATION` | `PT18H` | multiple of 3 h, at least 12 h |
| `--ensemble-size` | `CLARA_ENSEMBLE_SIZE` | `5` | number of ensemble members |
| `--ifs-boundary-path` | `CLARA_IFS_BOUNDARY_PATH` | (required) | IFS forecast zarr following the contract below (local or S3) |
| `--dini-root` | `CLARA_DINI_ROOT` | `s3://harmonie-zarr/dini/control/{analysis}/` | DINI forecast zarrs |
| `--workdir` | `CLARA_WORKDIR` | `inference_workdir` | working directory; outputs go in `outputs/` |
| `--artifact-path` | `CLARA_ARTIFACT_PATH` | `inference_artifact` | the inference artifact, including the checkpoint |
| `--skip-completed` | `CLARA_SKIP_COMPLETED` | `false` | skip the stages already completed in the workdir |
| `--output-s3-prefix` | `CLARA_OUTPUT_S3_PREFIX` | | also upload the outputs to `{prefix}/<analysis time>/`, replacing an earlier upload for that analysis time |
| `--input-s3-endpoint-url` | `CLARA_INPUT_S3_ENDPOINT_URL` | (AWS config) | S3 endpoint for reading DINI and IFS |
| `--input-aws-profile` | `CLARA_INPUT_AWS_PROFILE` | (AWS config) | AWS profile for reading DINI and IFS |
| `--output-s3-endpoint-url` | `CLARA_OUTPUT_S3_ENDPOINT_URL` | (AWS config) | S3 endpoint for the upload (needs `--output-s3-prefix`) |
| `--output-aws-profile` | `CLARA_OUTPUT_AWS_PROFILE` | (AWS config) | AWS profile for the upload (needs `--output-s3-prefix`) |
| `--debugger` | `CLARA_DEBUGGER` | | `ipdb` to debug the python steps on exceptions |
| `--log-level` | `CLARA_LOG_LEVEL` | `INFO` | loguru level |

### IFS boundary contract

The IFS boundary is not downloaded by the container. It is given as a zarr
(`--ifs-boundary-path`) in the layout the model was fine-tuned on. The
contract is checked by [ifs_boundary.py](src/clara_inference/ifs_boundary.py):

- **Dimensions:** `time` (forecast cycle) and `prediction_timedelta` (lead
  time, **every 6 h** from 0 h), plus `longitude` and `latitude` on the 0.25°
  WeatherBench2 grid. Either longitude convention and either latitude order
  is accepted. Pressure-level variables also have `level` (hPa).
- **Coverage:** the 400 km margin around the DANRA domain (lat 43.25–67.75,
  lon −22–44).
- **Variables**, with ERA5 names and units:
  - single level: `mean_sea_level_pressure`, `2m_temperature`,
    `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_pressure`;
  - pressure level: `geopotential`, `temperature`, `specific_humidity`,
    `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`, at
    100, 200, 400, 600, 700, 850, 925 and 1000 hPa.
- **Cycle selection:** the **latest 00/12 UTC cycle at or before the analysis
  time − 3 h**, which is the cycle neural-lam used in training. For example:
  - 06Z → 00Z;
  - 12Z → 00Z;
  - 18Z → 12Z;
  - 00Z → 12Z of the previous day.

  Its lead times must cover the forecast plus two 6 h steps.

**Point order.** The graph connects the boundary nodes by index, so the
boundary points must be in training order:
- longitude 0–360, increasing;
- within a longitude, latitude decreasing.

The IFS fields are therefore put on the artifact's boundary grid, which is in
that order, and the resulting boundary points are checked against the graph.

[dev-utils/make_ifs_boundary_zarr.py](dev-utils/make_ifs_boundary_zarr.py)
makes such a zarr from ECMWF open data. Open data is kept for a few days only.

### Outputs and conventions

The outputs are `<workdir>/outputs/single_levels.zarr` and
`pressure_levels.zarr`.
- **Grid:** the DANRA grid.
- **Times:** valid times T+6 h … T+duration in 3 h steps. The initial states
  are DINI at T+0 and T+3 h.
- **Ensemble:** an `ensemble_member` dimension, chunked per member and
  valid time.

The conventions are DANRA's, as for ANNA:
- winds are relative to the DANRA Lambert grid;
- `r` is a fraction (0–1);
- `lwavr0m`/`swavr0m` are net surface fluxes;
- `tw` is geometric vertical velocity.

The data carries CF grid-mapping metadata (`danra_projection`).

## Development

```bash
uv sync                          # the environment (exact versions in uv.lock)
uv run pre-commit install        # black (79), isort, flake8 on every commit
uv run pytest tests
```

`uv sync` installs neural-lam from the private `leifdenby/real-prob-lam`,
using your git credentials.

- [dev-utils/](dev-utils/):
  - `make_ifs_boundary_zarr.py`: an IFS boundary zarr from ECMWF open data;
  - `check_forecast_output.py`: sanity checks of a forecast. It reports the
    ensemble-mean RMSE and bias against DINI, and the ensemble spread, per
    valid time;
  - `plot_domain.py`: the domain figure above.
- Each pipeline step can also be run on its own, e.g.
  `uv run python -m clara_inference.create_inference_dataset --help`.

## Licence

The code is licensed under the MIT licence. The licence of the model
checkpoint and the inference package is to be decided.

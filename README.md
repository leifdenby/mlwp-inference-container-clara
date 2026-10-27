# mlwp-inference-container-clara

> **Status: work in progress.** The pipeline is being built; see
> [PLAN.md](PLAN.md) for what is done and what remains. Sections below
> describe the intended design and are marked *(planned)* where nothing is
> implemented yet.

This repository builds the inference container images for **CLARA**,
**C**RPS-**LA**M **R**apid **A**tmospheric ensembles. CLARA runs **CRPS-LAM**
operationally. "Rapid" refers to the model's main strength: each ensemble
member is generated in a single forward pass, about 39× faster than diffusion
baselines. CRPS-LAM is the probabilistic
machine-learning limited-area model of Larsson et al. (2025),
["CRPS-LAM: Probabilistic Regional Weather Forecasting with Continuous Ranked
Probability Score"](https://arxiv.org/abs/2510.09484).

- **Interior initial states:** **DINI**, regridded to the DANRA grid.
- **Boundary:** **IFS** forecasts.
- **Output:** an **ensemble forecast**.

**Model architecture:** a hybrid GNN/UNet backbone, neural-lam `crps` with
backbone `graph_unet2` and graph `feature_map_graph_198x148`. A noise vector
is injected at every step, so each ensemble member is drawn in a single
forward pass. It was trained with a CRPS loss.

**Training data:**
- **Interior:** DANRA, 2.5 km and 3-hourly.
- **Boundary:** first ERA5, then fine-tuned on IFS forecasts, on a 400 km ring
  around the DANRA domain.

Everything needed for that is in one Python project
([pyproject.toml](pyproject.toml), with the exact environment in `uv.lock`):
- **Inference pipeline:** `src/clara_inference/`, run by
  `python -m clara_inference`, which is the image's entrypoint.
- **Image build:** the [Containerfile](Containerfile) and
  [scripts/build_image.sh](scripts/build_image.sh).
- **Artifact tools:** the tools that build the model's *inference artifact*
  (configs, training statistics, grids, graph; see below), in
  `src/clara_inference/artifact/`.

The design follows
[`dmidk/mlwp-inference-container-anna`](https://github.com/dmidk/mlwp-inference-container-anna)
(ANNA, the deterministic DANRA model). Much of the code is shared with it,
including the option/environment convention, checkpoint fetching, DINI
regridding and the S3 upload.

## Model

The model is described in `artifact/configs/model.yaml` *(planned)*: the
checkpoint, the graph, the neural-lam arguments and the training neural-lam
config.

**Paper:** E. Larsson, J. Oskarsson, T. Landelius, F. Lindsten,
[arXiv:2510.09484](https://arxiv.org/abs/2510.09484).

**Checkpoint:** `CRPS_IFS_ALL_FINETUNE_AR12-train-crps-4x330-05_21_13-3613`.
- wandb project `real_prob_lam`, run `CRPS_IFS_ALL_FINETUNE_AR12`, epoch 92.
- Fine-tuned with 12-step rollouts and an IFS boundary, starting from
  `CRPS_IFS_ALL_FINETUNE-train-crps-4x330-05_20_08-5709`.
- About 251 M parameters, trained in `bf16-mixed`.

**Key settings:**

| Setting | Value |
|---|---|
| Model / backbone | `--model crps --backbone_model graph_unet2` |
| Widths | `--hidden_dim 330 --hidden_dim_grid 270` |
| Processor | `--processor_layers 4 --processor_model unet` |
| Noise | `--noise_dim 32 --noise_embedding linear` |
| Prediction | `--pred_residual` |
| Ensemble | `--ensemble_size` is configurable (default 5) |
| Time-delta options | none: no `--time_delta_enc_dim`, no `--dynamic_time_deltas` |

**Training code:** neural-lam from
[`leifdenby/real-prob-lam`](https://github.com/leifdenby/real-prob-lam)
(`neural-lam-dev-building-ml-lams/`), a fork of `joeloskarsson/neural-lam-dev`.

## Software

Pinned in [pyproject.toml](pyproject.toml) *(planned)*:

- **neural-lam:** `leifdenby/real-prob-lam` (subdirectory
  `neural-lam-dev-building-ml-lams`), branch `prob-model-inference`, at a
  pinned commit. This branch adds:
  - the bf16 and single-GPU inference fixes;
  - `NEURAL_LAM_STRICT_LOAD=1`, which makes loading a checkpoint fail instead
    of silently keeping randomly initialised weights for missing or
    mismatched parameters.
- **mllam-data-prep:** `Erik-Wikingsson/mllam-data-prep-real@0589539`
  (`rect-boundary`, with `latlon-domain-crop`), as used for training.
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

The trained model was set up with these configs:

| File | What |
|---|---|
| `danra_model_config_ifs_all.yaml` | the neural-lam config the model was trained with: state feature weights, clamping of `r` to [0, 1] |
| `danra_model1_config_ifs_finetune.yaml` | interior (DANRA) datastore, PT3H. 55 state features: pres_seasurface, t2m, u10m, v10m, pres0m, lwavr0m, swavr0m, and z/t/r/u/v/tw at 100–1000 hPa. Statics lsm, orography; forcings toa_radiation and time of day/year |
| `overlapping_ifs_all2_400km_model1_config.yaml` | boundary (IFS forecast) datastore. Forecast layout `analysis_time × elapsed_forecast_duration` with lead times every 6 h, 400 km margin (14,858 points of the 0.25° grid), 58 forcing features with ERA5 names, ERA5 statics |
| `danra_model1_config.yaml` | interior statistics (DANRA, 2016-01-01 → 2019-10-22) |
| `overlapping_era_400km_model1_config.yaml` | boundary statistics (ERA5, same period): the IFS boundary is normalised with ERA5 statistics (`overload_stats_path`) |

## Inference artifact *(planned)*

An **inference artifact** is a directory, shipped as a zip, that holds
everything needed to run a trained model's forecasts apart from the input data
and the software. See ANNA's README for the concept. CLARA's will have:

| Path | What |
|---|---|
| `configs/` | the training configs above and `model.yaml` |
| `stats/` | training statistics of the DANRA interior and ERA5 boundary datastores |
| `grids/` | the DANRA grid and statics (target grid for regridding DINI), and the ERA5 boundary statics on the 0.25° grid |
| `graphs/feature_map_graph_198x148/` | the model's graph, prebuilt (it is not rebuilt at run time) |
| `artifact.yaml`, `README.md` | provenance and description |
| `clara_model.ckpt` | the sanitized checkpoint: weights, `args` and `config` only, without optimizer state or pickled training datastores |

**The normalisation statistics are not in the checkpoint.** neural-lam
registers them as non-persistent buffers, and the training statistics only
existed on the training cluster. They are therefore **recomputed** with
`clara_inference.artifact.compute_stats`, over the training period, from:
- DANRA v0.5.0, for the interior;
- WeatherBench2 ERA5, for the boundary.

## Running a forecast *(planned)*

`python -m clara_inference` runs the whole pipeline. It is also installed as
`clara-inference`, which is the image's entrypoint. Each step runs as a
subprocess:

1. **`clara_inference.regrid_dini`:** regrids DINI to the DANRA grid at
   3-hourly steps.
2. **`clara_inference.create_inference_dataset`:** builds the inference
   datastores and the neural-lam config.
   - The interior is normalised with the DANRA training statistics.
   - The boundary uses the selected IFS cycle, normalised with the ERA5
     training statistics.
   - The artifact's graph is linked in.
3. **`neural_lam.train_model --eval test`:** runs the ensemble forecast with
   strict checkpoint loading.
4. **`clara_inference.convert_output`:** writes DANRA-like output datasets
   with an `ensemble_member` dimension.
5. **`clara_inference.upload_outputs`:** optionally uploads to S3.

### Configuration

Every command-line option `--some-option` can also be set with the
environment variable `CLARA_SOME_OPTION`. The command line wins over the
variable, which wins over the default. This is the same convention as ANNA
(`envargs.py`).

| Option | Environment variable | Default | |
|---|---|---|---|
| `--analysis-time` | `CLARA_ANALYSIS_TIME` | (required) | DINI analysis time, ISO8601, e.g. `2026-10-01T00:00Z` |
| `--forecast-duration` | `CLARA_FORECAST_DURATION` | `PT18H` | multiple of 3 h |
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
(`--ifs-boundary-path`) in the layout the model was fine-tuned on:

- **Dimensions:** `time` (forecast cycle) and `prediction_timedelta` (lead
  time, **every 6 h**), plus `longitude` and `latitude` on the 0.25°
  WeatherBench2 grid. Pressure-level variables also have `level` (hPa).
- **Coverage:** the 400 km margin around the DANRA domain.
- **Variables**, with ERA5 names and units:
  - single level: `mean_sea_level_pressure`, `2m_temperature`,
    `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_pressure`;
  - pressure level: `geopotential`, `temperature`, `specific_humidity`,
    `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`, at
    100, 200, 400, 600, 700, 850, 925 and 1000 hPa.
- **Cycle selection:** the latest cycle at or before the analysis time is
  used. Its lead times must cover the forecast.

`dev-utils/make_ifs_boundary_zarr.py` *(planned)* makes such a zarr from ECMWF
open data.

### Outputs and conventions

The outputs are `<workdir>/outputs/single_levels.zarr` and
`pressure_levels.zarr`.
- **Grid:** the DANRA grid.
- **Times:** valid times T+6 h … T+duration in 3 h steps. The initial states
  are DINI at T+0 and T+3 h.
- **Ensemble:** an `ensemble_member` dimension.

The conventions are DANRA's, as for ANNA:
- winds are relative to the DANRA Lambert grid;
- `r` is a fraction (0–1);
- `lwavr0m`/`swavr0m` are net surface fluxes;
- `tw` is geometric vertical velocity.

The data carries CF grid-mapping metadata (`danra_projection`).

## Development *(planned)*

```bash
uv sync
uv run pre-commit install
uv run pytest tests
```

## Licence

The code is licensed under the MIT licence. The licence of the model
checkpoint and the inference package is to be decided.

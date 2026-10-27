# PLAN: CLARA inference container

Living plan for `mlwp-inference-container-clara`; updated as work progresses.
Status: `[x]` done, `[~]` in progress, `[ ]` to do, `[?]` needs a decision.

## Goal

An operational inference container for **CLARA**, DMI's name for the
CRPS-LAM model (Larsson et al. 2025, [arXiv:2510.09484](https://arxiv.org/abs/2510.09484)).
CRPS-LAM is a probabilistic limited-area model: neural-lam `crps` with a
hybrid GNN + UNet backbone (`graph_unet2`) and noise injection, trained with
CRPS.

It works like
[`dmidk/mlwp-inference-container-anna`](https://github.com/dmidk/mlwp-inference-container-anna):
given an analysis time, it regrids DINI initial states to the DANRA grid and
takes an IFS boundary forecast zarr. It then runs an ensemble forecast and
writes DANRA-grid zarrs with an `ensemble_member` dimension, optionally
uploading them to S3.

| | |
|---|---|
| GitHub | `leifdenby/mlwp-inference-container-clara` (private) |
| Python package / CLI | `clara_inference` / `clara-inference` |
| Environment variables | `CLARA_*` |
| Image | `ghcr.io/leifdenby/mlwp-inference-container-clara` |
| Model source | `leifdenby/real-prob-lam`, subdir `neural-lam-dev-building-ml-lams` (private) |

## Decisions

| Topic | Decision |
|---|---|
| Repository | new anna-style repo |
| Normalisation stats | **recompute** from DANRA v0.5.0 and WB2 ERA5 over the training period (they're not in the checkpoint) |
| Interior time step | **3 h**, as in training (the local prototype wrongly used 1 h) |
| IFS boundary | **pre-made zarr path** (`--ifs-boundary-path`) following a written contract. Fetching IFS open data is a dev utility, not part of the run |

## Facts established about the trained model

From the checkpoint `CRPS_IFS_ALL_FINETUNE_AR12-train-crps-4x330-05_21_13-3613/last.ckpt`
and the training configs in `real-prob-lam/neural-lam-dev-building-ml-lams/configs/`:

**Checkpoint**
- Lightning 2.5.1, epoch 92, about 251 M parameters.
- 3.0 GB in total, including optimizer state.
- `hyper_parameters` contains `args`, `config` and a pickled `datastore_boundary` that points at Berzelius paths.

**Model args**
- `model=crps`, `backbone_model=graph_unet2`, `graph_name=feature_map_graph_198x148`
- `hidden_dim=330`, `hidden_dim_grid=270`, `hidden_dim_mesh_nodes=270`, `hidden_dim_edge=32`, `hidden_layers=1`, `processor_layers=4`, `processor_model=unet`
- `channel_mult=[1]*6`, `channel_mult_grid=[1,1,1]`, `attn_res=[10,18,36,70]`, `channel_mult_emb=4`, `channel_mult_noise=2`, `skip_type=cat`
- `noise_dim=32`, `noise_embedding=linear`, `pred_residual=True`, `interior_dim=127`
- all past/future forcing and boundary step counts = 1
- `precision=bf16-mixed`, `time_delta_enc_dim=None`, `dynamic_time_deltas=False`
- `ensemble_size=5`

**Interior**
- The full DANRA grid, 589×789 = 464,721 points, at PT3H.
- 55 state features: 7 single-level variables, plus z/t/r/u/v/tw at 100, 200, 400, 600, 700, 850, 925 and 1000 hPa.
- Static: lsm, orography.
- Forcing: toa_radiation and hour/day-of-year sin/cos.
- Training stats were computed over 2016-01-01 → 2019-10-22.

**Boundary**
- IFS forecasts in forecast layout (`analysis_time × elapsed_forecast_duration`), with lead times **every 6 h from 0 to 240 h** and cycles about every 12 h.
- 400 km margin with `include_interior_points: True`, giving 14,858 points from the 0.25° WB2 grid.
- 58 forcing features with ERA5 names; static is ERA5 `land_sea_mask` and `geopotential_at_surface`.
- Normalised with **ERA5** stats (`overload_stats_path` → `overlapping_era_400km_model1_config`, 2016-01-01 → 2019-10-22, PT6H).

**Normalisation stats are not in the checkpoint**
- The buffers are registered with `persistent=False`.
- The pickled stats datastore only references Berzelius zarrs.

**How neural-lam picks the boundary forecast**
- It uses the latest boundary `analysis_time` strictly before the model init time (`state_times[1]` = T+3 h), per `weather_dataset.py:511+`.
- So an IFS cycle at or before T works.

## Phase 0: prepare `real-prob-lam` (neural-lam source)

- [x] Commit the inference fixes on `prob-model-inference` (`47cc2fe`, not pushed):
  - bf16 `mesh_rep` dtype
  - single-device `auto` strategy
  - graphs loaded with `weights_only=False`
  - `persistent_workers` guard
  - fft2 CPU fallback
- [x] Add `NEURAL_LAM_STRICT_LOAD=1`, which makes `on_load_checkpoint` raise on missing or shape-mismatched parameters.
- [ ] Push `prob-model-inference` to `leifdenby/real-prob-lam`.
- [?] `real-prob-lam` is **private**, so the image build can't simply `pip install git+https://…`. Options:
  - (a) a build secret with a GitHub token
  - (b) vendor `neural-lam-dev-building-ml-lams` as a git submodule in the build context
  - (c) make `real-prob-lam` public

  Current lean: (b), the submodule, pinned to a commit.

## Phase 1: inference artifact (`src/clara_inference/artifact/`)

- [ ] `configs/` holds the training configs with Berzelius paths removed:
  - `danra_model1_config_ifs_finetune.yaml` (interior)
  - `overlapping_ifs_all2_400km_model1_config.yaml` (boundary)
  - `danra_model_config_ifs_all.yaml` (neural-lam config)
  - stats configs `danra_model1_config.yaml` and `overlapping_era_400km_model1_config.yaml`

  It also holds `model.yaml` (checkpoint URL/md5, graph name, `train_model_args`).
- [ ] `artifact/compute_stats.py` recomputes the training stats with mllam-data-prep:
  - DANRA v0.5.0 (ECMWF object store), PT3H, 2016-01-01 → 2019-10-22
  - WB2 ERA5 with the 400 km crop, PT6H, same period
  - keeps only the `*__train__*` variables

  It's run once, offline, and is heavy: roughly 11.4 k DANRA times × 55 features × 465 k points.
- [ ] `grids/`: the DANRA grid and statics, and the ERA5 boundary statics (lsm, z_sfc) on the WB2 0.25° grid subset.
- [ ] `graphs/feature_map_graph_198x148/`, from `graphs/feature_map_graph_198x148.zip`.
- [ ] `sanitize_checkpoint`:
  - drop `optimizer_states`, `lr_schedulers`, `loops`, `callbacks` and `hyper_parameters["datastore_boundary"]`
  - check that the `state_dict` is unchanged
  - (optionally) cast to bf16
- [ ] Publish the package and checkpoint as a GitHub release asset or on Zenodo.

## Phase 2: repo scaffold (copy and adapt from anna)

- [ ] `pyproject.toml`, `Containerfile`, `scripts/build_image.sh`, `scripts/run_inference_container.sh`
- [ ] `envargs.py` and `fetch_checkpoint.py`, unchanged
- [ ] `upload_outputs.py`, unchanged
- [ ] `run.py`, whose `Forecast` steps are:
  1. **`regrid_dini`**: interior only, bilinear, with wind rotation and `r/100`, at 3 h steps from T+0 → T+duration+extra.
  2. **`create_inference_dataset`**:
     - The interior datastore takes the training stats from the artifact.
     - The boundary datastore uses the selected IFS cycle in forecast layout, with the ERA5 stats overloaded.
     - It writes the neural-lam `config.yaml` and symlinks the artifact graph.
     - No configs are edited in place.
  3. **forecast**: `neural_lam.train_model --model crps … --ensemble_size N --eval test`, with strict loading and wandb disabled or offline.
  4. **`convert_output`**: writes single/pressure-level zarrs with an `ensemble_member` dimension.
  5. **`upload_outputs`**: optional.
- [ ] IFS boundary contract and validation:
  - ERA5 variable names, 8 pressure levels
  - 0.25° WB2 grid covering the 400 km margin
  - `time`, `prediction_timedelta` every 6 h
- [ ] `dev-utils/make_ifs_boundary_zarr.py`, adapted from `scripts/download_ifs_open_data_forecast.py` to produce **6-hourly** lead times.
- [ ] CI: pytest, pre-commit, and container build → ghcr.

## Phase 3: tests (CPU)

- [ ] Port anna's tests (envargs, run config, regrid, upload, fetch_checkpoint).
- [ ] New tests:
  - IFS contract and cycle selection
  - config feature order against the training configs
  - `model.yaml` args against the checkpoint hparams
  - `convert_output` with members

## Verification

- [ ] `uv run pytest`; build the image.
- [ ] The sanitized checkpoint loads with `NEURAL_LAM_STRICT_LOAD=1`.
- [ ] Recomputed stats are sane: same order as the subset stats, all std > 0.
- [ ] End-to-end on the DGX Spark: no NaNs, plausible values, ensemble spread > 0 that grows with lead time, sensible CRPS against DINI; record runtime and memory.

## Log

- 2026-10-27: Explored anna and the local prototype; decisions above. Phase 0 commit `47cc2fe`. From the checkpoint: IFS lead step is 6 h, and the stats are not recoverable from it.

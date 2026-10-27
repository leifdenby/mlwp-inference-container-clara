# PLAN: CLARA inference container

Living plan for `mlwp-inference-container-clara`; updated as work progresses.
Status: `[x]` done, `[~]` in progress, `[ ]` to do, `[?]` needs a decision.

## Goal

An operational inference container for **CLARA** (**C**RPS-**LA**M **R**apid
**A**tmospheric ensembles), DMI's name for the CRPS-LAM model (Larsson et al.
2025, [arXiv:2510.09484](https://arxiv.org/abs/2510.09484)). CRPS-LAM is a
probabilistic limited-area model: neural-lam `crps` with a hybrid GNN + UNet
backbone (`graph_unet2`) and noise injection, trained with CRPS.

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
| Model source | `leifdenby/real-prob-lam`, subdir `neural-lam-dev-building-ml-lams` (private), pinned at `d96b972` |

## Decisions

| Topic | Decision |
|---|---|
| Repository | new anna-style repo; both repos private, no git submodules |
| Normalisation stats | **recompute** from DANRA v0.5.0 and WB2 ERA5 over the training period (they're not in the checkpoint) |
| Interior time step | **3 h**, as in training (the local prototype wrongly used 1 h) |
| IFS boundary | **pre-made zarr path** (`--ifs-boundary-path`) following a written contract. Fetching IFS open data is a dev utility, not part of the run |
| Private dependencies in the image | GitHub token as a BuildKit secret (`github_token`), never a build arg or layer |
| Checkpoint/package hosting | release assets of the private repo, downloaded through the GitHub API with the token (no Zenodo) |

## Facts established about the trained model

From the checkpoint `CRPS_IFS_ALL_FINETUNE_AR12-train-crps-4x330-05_21_13-3613/last.ckpt`
and the training configs in `real-prob-lam/neural-lam-dev-building-ml-lams/configs/`:

**Checkpoint**
- Lightning 2.5.1, epoch 92, about 251 M parameters.
- 3.0 GB in total, including optimizer state.
- `hyper_parameters` contains `args`, `config` and a pickled `datastore_boundary` that points at Berzelius paths.

**Model args**
- The args that matter and differ from `train_model`'s defaults are all in `artifact/configs/model.yaml`, checked against the parser.
- `model=crps`, `backbone_model=graph_unet2`, `graph_name=feature_map_graph_198x148`
- `hidden_dim=330`, `hidden_dim_grid=270`, `hidden_dim_mesh_nodes=270`, `hidden_dim_edge=32`
- `channel_mult=[1]*6`, `attn_res=[10,18,36,70]`, `noise_dim=32`, `noise_embedding=linear`, `pred_residual`, `precision=bf16-mixed`
- `interior_dim=127` is computed by the model, not an argument.
- The local prototype passed no `hidden_dim_mesh_nodes`.

**Interior**
- The full DANRA grid, 589×789 = 464,721 points, at PT3H.
- 55 state features: 7 single-level variables, plus z/t/r/u/v/tw at 100, 200, 400, 600, 700, 850, 925 and 1000 hPa.
- Static: lsm, orography.
- Forcing: toa_radiation and hour/day-of-year sin/cos.

**Boundary**
- IFS forecasts in forecast layout, 00/12 UTC cycles, lead times **every 6 h from 0 to 240 h**.
- 400 km margin with interior points included: 14,858 points.
- 58 forcing features with ERA5 names; static is ERA5 lsm and surface geopotential.
- Normalised with **ERA5** stats.

**Boundary point order**
- The graph connects boundary nodes by index, so point order matters.
- In training, points were ordered by longitude (0..360, increasing), then latitude (decreasing).
- The local prototype's IFS data had latitude increasing. It rebuilt the graph for its own order, so it was self-consistent: the graph is the same up to relabelling of the boundary nodes.
- CLARA ships the training graph, puts the IFS data on a fixed grid in training order, and checks the boundary points against the graph's boundary nodes (max 0.13 m).

**Cycle selection**
- neural-lam takes the latest boundary cycle strictly before T+3 h, stepping back one cycle if that leaves less than one 6 h step.
- With 00/12 UTC cycles in training, this means the latest 00/12 cycle at or before T−3 h. CLARA gives neural-lam exactly that one cycle.

**Normalisation stats**
- They are not in the checkpoint: the buffers are `persistent=False`, and the pickled stats datastore only references Berzelius zarrs.
- mllam-data-prep's `diff_std` is the std of **second** time differences: `calc_stats` replaces the dataset by its diff for each `diff_*` op. `diff_mean` is the mean of first differences. Verified on a local datastore to 1e-15.

**Placeholder stats from a few days are unusable**
- `day_of_year_sin/cos` then have std ≈ 0, which gives standardised values of about −1600σ and a diverging forecast.

## Phase 0: prepare `real-prob-lam` (neural-lam source)

- [x] Commit the inference fixes on `prob-model-inference`: bf16 `mesh_rep` dtype, single-device `auto` strategy, graph `weights_only=False`, `persistent_workers` guard, fft2 CPU fallback.
- [x] `NEURAL_LAM_STRICT_LOAD=1`: `on_load_checkpoint` raises on missing or shape-mismatched parameters.
- [x] Push `prob-model-inference` (`47cc2fe`).

## Phase 1: inference artifact (`src/clara_inference/artifact/`)

- [x] `artifact/configs/`:
  - the 5 training configs with Berzelius paths replaced by public sources or placeholders
  - `model.yaml` with the checkpoint, graph and `train_model_args`
- [x] `compute_stats`:
  - mllam-data-prep extracts blocks (4 days, overlapping by 2 steps); the statistics are combined exactly in float64.
  - mllam-data-prep semantics, including second-difference `diff_std`.
  - Matches mllam-data-prep on real data to 2.5e-8 of each feature's spread; unit tests cover the block combination.
  - `--from-processed` for placeholder stats.
- [x] Parallel segments (`--workers`) and a local chunk cache (`--cache-dir`).
  - DANRA v0.5.0 is 3-hourly, float64, with time chunks of 256 steps (32 days). Without a cache, every block re-downloaded whole chunks.
  - With the cache, everything is downloaded once: about 540 GB for DANRA. WB2 ERA5 is one global chunk per time step, about 1 TB.
  - This machine has about 60 MB/s over WiFi, so expect hours.
- [~] Full training-period stats run, in progress:
  - DANRA from the ECMWF object store and WB2 ERA5 from GCS, 2016-01-01 → 2019-10-22.
  - Logs: `artifact/build/stats-{interior,boundary}.log`.
- [x] `assemble`:
  - configs, stats datastores, graph from the training zip, DANRA grid from the object store
  - boundary grid derived from the graph's boundary nodes, with ERA5 statics from WB2
  - sanitized checkpoint, `artifact.yaml`, README
  - `layout.check_artifact`
- [x] `sanitize_checkpoint`:
  - drops the training state and the pickled datastore
  - stores `args`/`config` as plain dicts, so the checkpoint loads with `weights_only=True`
  - weights checked identical: 1,397 tensors, 251.5 M values, 1.0 GB
- [ ] Assemble the real artifact with the full stats; publish the package zip and checkpoint as release assets; fill in `model.yaml`'s url/md5.

## Phase 2: pipeline and repo

- [x] Import the template files from anna `a7033c2` unmodified, then rename to `clara_inference`/`CLARA_`.
- [x] README with name, domain plot, configs, IFS contract and options; PLAN.md.
- [x] `pyproject.toml`: neural-lam from real-prob-lam@`47cc2fe`, the training mllam-data-prep pin, torch 2.12 cu130; `uv.lock`.
- [x] `regrid_dini`: interior only, 3-hourly.
- [x] `storage.open_zarr`: anonymous S3 reads without AWS credentials (DINI's bucket is public).
- [x] `ifs_boundary`: contract check, cycle selection, lead times, boundary grid in training order, point-order check against the graph.
- [x] `create_inference_dataset`:
  - interior and boundary datastores
  - checks that the stats features match by position
  - graph link and neural-lam config
- [x] `run`: `--ensemble-size`, required `--ifs-boundary-path`, strict loading, no graph build.
- [x] `convert_output`: `ensemble_member` dimension, CLARA metadata, chunked per member and time.
- [x] `fetch_checkpoint`: `GITHUB_TOKEN` for API release assets; the token is dropped on cross-host redirects.
- [x] `dev-utils/make_ifs_boundary_zarr.py` (open data, 00/12 cycle, 6-hourly).
- [x] `dev-utils/check_forecast_output.py`: ensemble-mean RMSE/bias against DINI, and spread.
- [x] `Containerfile`:
  - token secret
  - `--no-sources` plus `scripts/container_pins.py` (git pins from pyproject), so torch is the base image's
  - artifact from `local` or `release`
- [x] `scripts/build_image.sh` (token from `gh auth token`), `scripts/run_inference_container.sh`.
- [x] CI: pytest and container build/publish with the `REAL_PROB_LAM_TOKEN` secret and `PACKAGE_URL`/`PACKAGE_MD5` variables.
- [ ] Build the image locally and run a forecast in it.

## Phase 3: tests (CPU)

- [x] Ported anna's tests: envargs, run config including the README option table, regrid, upload, fetch_checkpoint including token handling.
- [x] New tests: IFS cycle selection, lead times, contract, reordering onto the boundary grid, point-order check, `convert_output` with members, `compute_stats` block combination.

## Verification

- [x] `uv run pytest`: all pass.
- [x] The sanitized checkpoint loads with strict loading: 1,397 parameters, 0 missing, 0 mismatched.
- [x] End-to-end mechanics on the DGX Spark: 2026-10-26T06Z +18 h, 5 members, about 2.5 min for forecast and conversion; the boundary order check passed.
- [ ] End-to-end with the real stats: no NaNs, plausible values, spread > 0 and growing, RMSE against DINI of the size of short-range forecast differences. The placeholder-stats run diverges, as expected.
- [ ] Image build and container run.

## Log

- 2026-10-27:
  - Explored anna and the local prototype; made the decisions above.
  - Phase 0 commit `47cc2fe`. From the checkpoint: the IFS lead step is 6 h, and the stats are not recoverable.
  - Found the boundary point-order issue and the second-difference `diff_std`.
  - Pipeline runs end to end with placeholder stats; started the full stats runs.

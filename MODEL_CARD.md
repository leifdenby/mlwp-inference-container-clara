# CLARA model card

<!-- generated from artifact/model-card.yaml by `python -m clara_inference.model_card`: edit that file instead -->

**CLARA** (CRPS-LAM Rapid Atmospheric ensembles). A probabilistic limited-area model for the DANRA domain (Denmark and surroundings, 2.5 km). Each ensemble member is a rollout of the CRPS-LAM model in 3 h steps from two DINI initial states, with lateral boundary conditions from an IFS forecast. The members differ through noise injected into the model, not through perturbed initial states.

## Model

- **Paper:** CRPS-LAM: Probabilistic Regional Weather Forecasting with Continuous Ranked Probability Score, Erik Larsson, Joel Oskarsson, Tomas Landelius, Fredrik Lindsten, [arXiv:2510.09484](https://arxiv.org/abs/2510.09484)
- **Architecture:** the crps model of neural-lam with a graph_unet2 backbone, 251.5 M parameters, trained with a CRPS objective (see the paper). This checkpoint is the fine-tune with an IFS forecast boundary: 5-member ensembles over 12 autoregressive steps, 93 epochs
- **Training data:**
  - interior: DANRA v0.5.0 reanalysis, 3-hourly
  - boundary: IFS forecasts, 00/12 UTC cycles, 6-hourly lead times
  - period: 2016-01-01 to 2019-10-22
  - statistics: the interior is normalised with DANRA statistics, the IFS boundary with ERA5 statistics, both over the training period
- **Intended use:** Research and experimental short-range ensemble forecasts over the DANRA domain from DINI analyses. Not an operational product: the forecasts have not been verified beyond example cases.

## Time, ensemble and grids

- **Step:** PT3H
- **Initial states:** two DINI states, at the analysis time T and T+3 h (DINI's own 3 h forecast)
- **Outputs:** valid times T+6 h to T+duration in 3 h steps
- **Boundary:** the latest 00/12 UTC IFS cycle at or before T-3 h, with lead times every 6 h
- **Instantaneous:** all fields are instantaneous values at the valid time (no accumulations or time means)
- **Ensemble:** `ensemble_member` dimension (0, 1, ...). Members are exchangeable: their numbering has no meaning. Default 5 members.
- **Interior grid:** the DANRA grid: Lambert conformal, 2.5 km, 589 x 789 points (CF grid mapping `danra_projection` in the output files)
- **Boundary grid:** 14,858 points of the 0.25 deg regular latitude-longitude grid within 400 km of the interior, including the points inside the interior

## Output files

- `single_levels.zarr`: the height- and single-level state variables, dims (ensemble_member, time, y, x)
- `pressure_levels.zarr`: the pressure-level state variables, dims (ensemble_member, time, pressure, y, x)
- The output variables carry `units` and the grid mapping only; their `long_name` and `standard_name` are planned to come from this card.

## Reading the tables

- **CF standard name:** from the CF standard name table, version 95. (none) means that no standard name exists for the variable.
- **CF:** whether the variable follows CF as stored (in the output files for the state, in the datastores for the other groups). **yes**: the standard name and units describe it. **partial**: a standard name applies, but something about the variable as stored differs. **no**: there is no standard name. The note says why, or where the source datasets' own metadata differ.
- **Source:** where the operational inputs come from, and how they are converted.
- **Loss weight:** the state feature's weight in the training loss (per level for pressure levels).
- Tables are grouped by level type:
  - Height levels: at a fixed height above ground (`levels`, in m)
  - Single levels: at the surface, at mean sea level, at the top of the atmosphere or without a vertical coordinate
  - Pressure levels: on pressure levels (`levels`, in hPa)

## Interior state (initial states and forecast)

The 55 prognostic features on the DANRA grid: the model's inputs at the two initial times and its outputs at every valid time. Trained on DANRA; operationally the initial states come from DINI, regridded to the DANRA grid (bilinear, in DINI's projection).

### Height levels

Output file: `single_levels.zarr`.

| Variable | Description | Units | Levels | CF standard name | CF | Source | Loss weight | Notes |
|---|---|---|---|---|---|---|---|---|
| `t2m` | air temperature at 2 m | `K` | 2 m | `air_temperature` | **partial**: CF needs a scalar `height` coordinate (2 m), which the output files don't have yet | DINI t2m | 1 |  |
| `u10m` | wind component along the DANRA grid x-axis at 10 m | `m s-1` | 10 m | `x_wind` | **partial**: CF needs a scalar `height` coordinate (10 m), which the output files don't have yet. DINI labels its (also grid-relative) winds eastward_wind, which is wrong | DINI u10m/v10m, rotated from the DINI grid to the DANRA grid | 1 | relative to the DANRA Lambert grid, not eastward: rotate with the grid's angle to get east/north components |
| `v10m` | wind component along the DANRA grid y-axis at 10 m | `m s-1` | 10 m | `y_wind` | **partial**: CF needs a scalar `height` coordinate (10 m), which the output files don't have yet. DINI labels its (also grid-relative) winds northward_wind, which is wrong | DINI u10m/v10m, rotated from the DINI grid to the DANRA grid | 1 | relative to the DANRA Lambert grid, not northward |

### Single levels

Output file: `single_levels.zarr`.

| Variable | Description | Units | Levels | CF standard name | CF | Source | Loss weight | Notes |
|---|---|---|---|---|---|---|---|---|
| `pres_seasurface` | air pressure at mean sea level | `Pa` |  | `air_pressure_at_mean_sea_level` | yes | DINI pres_seasurface | 1 |  |
| `pres0m` | surface air pressure | `Pa` |  | `surface_air_pressure` | yes: DANRA labels it air_pressure | DINI pres0m | 1 |  |
| `lwavr0m` | net long-wave radiation flux at the surface | `W m-2` |  | `surface_net_downward_longwave_flux` | yes: DANRA has no standard name; DINI's long_name says downwelling | DINI lwavr0m | 1 | positive downwards, so mostly negative; close to zero under low cloud |
| `swavr0m` | net short-wave radiation flux at the surface | `W m-2` |  | `surface_net_downward_shortwave_flux` | yes: DANRA has no standard name; DINI labels it surface_downwelling_shortwave_flux, but it is the same net flux | DINI swavr0m | 1 | positive downwards. Can be negative at night in the forecasts (not physical; see limitations) |

### Pressure levels

Output file: `pressure_levels.zarr`.

| Variable | Description | Units | Levels | CF standard name | CF | Source | Loss weight | Notes |
|---|---|---|---|---|---|---|---|---|
| `z` | geopotential | `m2 s-2` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `geopotential` | yes | DINI z | 0.125 |  |
| `t` | air temperature | `K` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `air_temperature` | yes | DINI t | 0.125 |  |
| `r` | relative humidity | `1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `relative_humidity` | yes: a fraction (0-1). DANRA's units say %, but its values are fractions; DINI's are in % | DINI r, divided by 100 | 0.125 | can be outside [0, 1] in the forecasts (see limitations) |
| `u` | wind component along the DANRA grid x-axis | `m s-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `x_wind` | yes: DINI labels its (also grid-relative) winds eastward_wind | DINI u/v, rotated from the DINI grid to the DANRA grid | 0.125 | relative to the DANRA Lambert grid, not eastward |
| `v` | wind component along the DANRA grid y-axis | `m s-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `y_wind` | yes: DINI labels its (also grid-relative) winds northward_wind | DINI u/v, rotated from the DINI grid to the DANRA grid | 0.125 | relative to the DANRA Lambert grid, not northward |
| `tw` | upward air velocity (geometric vertical velocity) | `m s-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `upward_air_velocity` | yes | DINI tw | 0.125 | in m s-1, unlike the IFS boundary's vertical velocity (Pa s-1) |

## Interior forcings

Computed from the valid time (and position) at every step, on the DANRA grid.

### Single levels

| Variable | Description | Units | Levels | CF standard name | CF | Source | Notes |
|---|---|---|---|---|---|---|---|
| `toa_radiation` | incoming short-wave radiation at the top of the atmosphere | `W m-2` |  | `toa_incoming_shortwave_flux` | yes | computed (mllam-data-prep): solar constant 1366 W m-2 times the cosine of the solar zenith angle (0 at night), no atmosphere |  |
| `hour_of_day_sin` | sine of the hour of the day (2 pi * hour / 24, UTC) | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time (UTC) |  |
| `hour_of_day_cos` | cosine of the hour of the day | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time (UTC) |  |
| `day_of_year_sin` | sine of the day of the year (2 pi * day / 366) | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time |  |
| `day_of_year_cos` | cosine of the day of the year | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time |  |

## Interior static fields

From DANRA, shipped in the inference artifact.

### Single levels

| Variable | Description | Units | Levels | CF standard name | CF | Source | Notes |
|---|---|---|---|---|---|---|---|
| `lsm` | land-sea mask (1 land, 0 sea) | `1` |  | `land_binary_mask` | yes | DANRA v0.5.0 (artifact) |  |
| `orography` | surface geopotential | `m2 s-2` |  | `surface_geopotential` | yes: despite its name, the surface geopotential (DANRA labels it geopotential), not surface_altitude in m | DANRA v0.5.0 (artifact) |  |

## Boundary forcings (IFS)

The lateral boundary: an IFS forecast on the boundary points (0.25 deg), with ERA5 names and units, plus the same computed forcings as the interior. Normalised with ERA5 statistics. Operationally from ECMWF open data (dev-utils/make_ifs_boundary_zarr.py, IFS parameter given in the source column).

### Height levels

| Variable | Description | Units | Levels | CF standard name | CF | Source | Notes |
|---|---|---|---|---|---|---|---|
| `2m_temperature` | air temperature at 2 m | `K` | 2 m | `air_temperature` | yes | IFS 2t |  |
| `10m_u_component_of_wind` | eastward wind at 10 m | `m s-1` | 10 m | `eastward_wind` | yes | IFS 10u | earth-relative, unlike the interior winds |
| `10m_v_component_of_wind` | northward wind at 10 m | `m s-1` | 10 m | `northward_wind` | yes | IFS 10v | earth-relative, unlike the interior winds |

### Single levels

| Variable | Description | Units | Levels | CF standard name | CF | Source | Notes |
|---|---|---|---|---|---|---|---|
| `mean_sea_level_pressure` | air pressure at mean sea level | `Pa` |  | `air_pressure_at_mean_sea_level` | yes | IFS msl |  |
| `surface_pressure` | surface air pressure | `Pa` |  | `surface_air_pressure` | yes | IFS sp |  |
| `toa_radiation` | incoming short-wave radiation at the top of the atmosphere | `W m-2` |  | `toa_incoming_shortwave_flux` | yes | computed at the boundary points (as for the interior) |  |
| `hour_of_day_sin` | sine of the hour of the day | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time (UTC) |  |
| `hour_of_day_cos` | cosine of the hour of the day | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time (UTC) |  |
| `day_of_year_sin` | sine of the day of the year | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time |  |
| `day_of_year_cos` | cosine of the day of the year | `1` |  | (none) | **no**: a model feature, not a physical quantity | computed from the valid time |  |

### Pressure levels

| Variable | Description | Units | Levels | CF standard name | CF | Source | Notes |
|---|---|---|---|---|---|---|---|
| `geopotential` | geopotential | `m2 s-2` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `geopotential` | yes | IFS gh (geopotential height), multiplied by g = 9.80665 m s-2 |  |
| `temperature` | air temperature | `K` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `air_temperature` | yes | IFS t |  |
| `specific_humidity` | specific humidity | `kg kg-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `specific_humidity` | yes | IFS q | the interior has relative humidity instead |
| `u_component_of_wind` | eastward wind | `m s-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `eastward_wind` | yes | IFS u | earth-relative, unlike the interior winds |
| `v_component_of_wind` | northward wind | `m s-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `northward_wind` | yes | IFS v | earth-relative, unlike the interior winds |
| `vertical_velocity` | vertical velocity in pressure coordinates (omega) | `Pa s-1` | 100, 200, 400, 600, 700, 850, 925, 1000 hPa | `lagrangian_tendency_of_air_pressure` | yes | IFS w | positive downwards (increasing pressure), in Pa s-1, unlike the interior's tw (m s-1, positive upwards) |

## Boundary static fields (ERA5)

From ERA5 (WeatherBench2), shipped in the inference artifact.

### Single levels

| Variable | Description | Units | Levels | CF standard name | CF | Source | Notes |
|---|---|---|---|---|---|---|---|
| `land_sea_mask` | land area fraction | `1` |  | `land_area_fraction` | yes: a fraction (0-1), though WeatherBench2 labels it land_binary_mask | ERA5 (WeatherBench2, artifact) |  |
| `geopotential_at_surface` | surface geopotential | `m2 s-2` |  | `surface_geopotential` | yes | ERA5 (WeatherBench2, artifact) |  |

## CF summary

- **Interior state (initial states and forecast):** 10 yes; 3 partial (`t2m`, `u10m`, `v10m`)
- **Interior forcings:** 1 yes; 4 no (`hour_of_day_sin`, `hour_of_day_cos`, `day_of_year_sin`, `day_of_year_cos`)
- **Interior static fields:** 2 yes
- **Boundary forcings (IFS):** 12 yes; 4 no (`hour_of_day_sin`, `hour_of_day_cos`, `day_of_year_sin`, `day_of_year_cos`)
- **Boundary static fields (ERA5):** 2 yes

## Not available

- **precipitation:** no precipitation of any kind
- **cloud cover:** no cloud variables; relative humidity on pressure levels and the net long-wave flux at the surface are rough proxies
- **near-surface humidity:** no 2 m humidity or dew point; the lowest humidity is r at 1000 hPa
- **wind gusts:** only mean 10 m winds
- **visibility, fog:** none
- **snow:** no snow depth or snowfall
- **top-of-atmosphere outgoing radiation:** the radiation fluxes are at the surface only
- **skin or soil temperature:** none

## Limitations

- The model doesn't keep its outputs within physical ranges: relative humidity can be outside [0, 1] and net short-wave radiation negative at some grid points. The training config's output clamping isn't applied by the crps model. `--clip-to-physical-ranges` clips them in the output files (off by default).
- The ensemble spread is somewhat smaller than the ensemble-mean error against DINI in the example cases.
- There are no precipitation, cloud, near-surface humidity, gust, visibility or snow variables (see Not available).
- The initial states come from DINI, whereas the model was trained on DANRA states; differences between the two (e.g. in the radiation fields) may affect the first steps.

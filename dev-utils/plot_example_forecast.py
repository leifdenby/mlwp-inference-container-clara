"""
README figures of a CLARA ensemble forecast written by `python -m
clara_inference`, compared with the DINI forecast it was started from
(regridded to the DANRA grid, `inputs/interior_*_levels.zarr` in the
workdir):

- `clara_example_t2m_mslp.png`: ensemble-mean t2m with mslp contours, the
  ensemble spread of t2m and DINI's t2m, at the last valid time
- `clara_example_members.png`: 10 m wind speed of each member and of the
  ensemble mean, at the last valid time
- `clara_example_scores.png`: ensemble-mean rmse against DINI and ensemble
  spread by lead time (the scores of `check_forecast_output.py`)

Usage (cartopy downloads coastlines on first use):

    uv run --with matplotlib python dev-utils/plot_example_forecast.py \\
        <workdir> docs
"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from check_forecast_output import MEMBER, _scores

workdir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

# as in plot_domain.py: categorical slots 1 and 2, text and surface tokens
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d6d5d0"

# the DANRA projection (extra.projection in the interior datastore config)
DANRA_CRS = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=(56.7, 56.7),
    globe=ccrs.Globe(semimajor_axis=6367470.0, semiminor_axis=6367470.0),
)

sl = xr.open_zarr(workdir / "outputs" / "single_levels.zarr")
pl = xr.open_zarr(workdir / "outputs" / "pressure_levels.zarr")
sl_dini = xr.open_zarr(workdir / "inputs" / "interior_single_levels.zarr")
pl_dini = xr.open_zarr(workdir / "inputs" / "interior_pressure_levels.zarr")

analysis = np.datetime64(sl.attrs["analysis_time"])
valid = sl.time.values[-1]
lead = int((valid - analysis) / np.timedelta64(1, "h"))
n_members = sl.sizes[MEMBER]
when = (
    f"analysis {str(analysis)[:13]}Z, +{lead} h "
    f"(valid {str(valid)[:13]}Z), {n_members} members"
)


def _map_axes(fig, rect):
    ax = fig.add_axes(rect, projection=DANRA_CRS, facecolor=SURFACE)
    ax.set_extent(
        [sl.x.min(), sl.x.max(), sl.y.min(), sl.y.max()], crs=DANRA_CRS
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=INK, linewidth=0.4
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    return ax


def _field(ax, da, **kwargs):
    return ax.pcolormesh(
        sl.x.values,
        sl.y.values,
        da.values,
        transform=DANRA_CRS,
        shading="auto",
        rasterized=True,
        **kwargs,
    )


def _title(ax, text):
    ax.set_title(text, color=INK, fontsize=10, loc="left")


def _colorbar(fig, mappable, rect, label):
    cb = fig.colorbar(mappable, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_label(label, color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    return cb


def _suptitle(fig, text):
    fig.text(0.02, 0.97, text, color=INK, fontsize=12, ha="left", va="top")


# NB: no bbox_inches="tight", which crops cartopy maps wrongly with the
# matplotlib version in the neural-lam environment
def _save(fig, name):
    fp = out_dir / name
    fig.savefig(fp, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {fp}")


# 1. t2m and mslp: ensemble mean, spread, DINI
t2m = sl.t2m.sel(time=valid).load() - 273.15
t2m_dini = sl_dini.t2m.sel(time=valid).load() - 273.15
mslp_mean = sl.pres_seasurface.sel(time=valid).mean(MEMBER).load() / 100
mslp_dini = sl_dini.pres_seasurface.sel(time=valid).load() / 100
t2m_mean = t2m.mean(MEMBER)
t2m_spread = t2m.std(MEMBER)

fig = plt.figure(figsize=(15, 5.4), facecolor=SURFACE)
vmin, vmax = np.percentile(t2m_dini, [1, 99])
isobars = np.arange(940, 1060, 4)
panels = [
    ("CLARA ensemble mean: t2m, mslp (hPa)", t2m_mean, mslp_mean),
    ("CLARA ensemble spread of t2m", t2m_spread, None),
    ("DINI: t2m, mslp (hPa)", t2m_dini, mslp_dini),
]
for i, (title, da, mslp) in enumerate(panels):
    ax = _map_axes(fig, [0.02 + i * 0.33, 0.2, 0.3, 0.66])
    if i == 1:
        im_spread = _field(ax, da, cmap="Blues", vmin=0)
    else:
        im_t = _field(ax, da, cmap="viridis", vmin=vmin, vmax=vmax)
    if mslp is not None:
        cs = ax.contour(
            sl.x.values,
            sl.y.values,
            mslp.values,
            levels=isobars,
            colors=SURFACE,
            linewidths=0.8,
            transform=DANRA_CRS,
        )
        ax.clabel(cs, fontsize=7, colors=SURFACE, fmt="%d")
    _title(ax, title)
_colorbar(fig, im_t, [0.02, 0.12, 0.3, 0.03], "2 m temperature (°C)")
_colorbar(fig, im_spread, [0.35, 0.12, 0.3, 0.03], "ensemble std of t2m (K)")
_colorbar(fig, im_t, [0.68, 0.12, 0.3, 0.03], "2 m temperature (°C)")
_suptitle(fig, f"CLARA example forecast: {when}")
_save(fig, "clara_example_t2m_mslp.png")

# 2. 10 m wind speed of each member and of the ensemble mean
wind = np.hypot(sl.u10m.sel(time=valid), sl.v10m.sel(time=valid)).load()
wind_mean = np.hypot(
    sl.u10m.sel(time=valid).mean(MEMBER), sl.v10m.sel(time=valid).mean(MEMBER)
).load()
n_cols = 3
n_rows = int(np.ceil((n_members + 1) / n_cols))
fig = plt.figure(figsize=(13, 3.3 * n_rows + 0.9), facecolor=SURFACE)
height = 0.84 / n_rows
vmax = float(np.percentile(wind, 99.5))
stamps = [(f"member {int(m)}", wind.sel({MEMBER: m})) for m in wind[MEMBER]]
stamps.append(("ensemble mean (of u and v)", wind_mean))
for i, (title, da) in enumerate(stamps):
    row, col = divmod(i, n_cols)
    ax = _map_axes(
        fig,
        [0.02 + col * 0.33, 0.93 - (row + 1) * height + 0.02, 0.3,
         height - 0.05],
    )
    im = _field(ax, da, cmap="YlGnBu", vmin=0, vmax=vmax)
    _title(ax, title)
_colorbar(fig, im, [0.35, 0.075, 0.3, 0.018], "10 m wind speed (m/s)")
_suptitle(fig, f"CLARA example forecast, members: {when}")
_save(fig, "clara_example_members.png")

# 3. rmse against DINI and spread by lead time
common = np.intersect1d(sl.time.values, sl_dini.time.values)
scores = [
    ("t2m (K)", sl.t2m, sl_dini.t2m, 1),
    ("mslp (hPa)", sl.pres_seasurface, sl_dini.pres_seasurface, 100),
    ("u10m (m/s)", sl.u10m, sl_dini.u10m, 1),
    ("v10m (m/s)", sl.v10m, sl_dini.v10m, 1),
    ("t 850 hPa (K)", pl.t.sel(pressure=850), pl_dini.t.sel(pressure=850), 1),
    (
        "z 400 hPa (m²/s²)",
        pl.z.sel(pressure=400),
        pl_dini.z.sel(pressure=400),
        1,
    ),
]
fig, axes = plt.subplots(
    2, 3, figsize=(12, 6.4), facecolor=SURFACE, sharex=True
)
for ax, (title, da, da_dini, scale) in zip(axes.flat, scores):
    rmse, _, spread = _scores(da.load(), da_dini.load(), common)
    hours = (rmse.time.values - analysis) / np.timedelta64(1, "h")
    hours_spread = (spread.time.values - analysis) / np.timedelta64(1, "h")
    ax.set_facecolor(SURFACE)
    ax.plot(hours, rmse / scale, color=SERIES_1, lw=2, marker="o", ms=5,
            label="ensemble-mean rmse vs DINI")
    ax.plot(hours_spread, spread / scale, color=SERIES_2, lw=2, marker="o",
            ms=5, label="ensemble spread")
    ax.set_ylim(bottom=0)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.5)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_edgecolor(GRID)
for ax in axes[-1]:
    ax.set_xlabel("lead time (h)", color=INK_2, fontsize=9)
    ax.set_xticks(hours)
axes.flat[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.1, hspace=0.35)
_suptitle(
    fig,
    "CLARA example forecast: ensemble-mean rmse against DINI and ensemble "
    f"spread over the domain\nanalysis {str(analysis)[:13]}Z, {n_members} "
    "members",
)
_save(fig, "clara_example_scores.png")

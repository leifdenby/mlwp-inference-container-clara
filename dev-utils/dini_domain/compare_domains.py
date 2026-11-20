"""
Compare a CLARA forecast on the DINI domain (`make_dini_domain_artifact.py`)
with the standard CLARA forecast (DANRA domain) from the same initial time
and IFS cycle: both against the DINI forecast they were started from
(regridded to each grid, `inputs/` of the workdirs), as ensemble-mean RMSE
and ensemble spread per lead time,

- standard CLARA, over the DANRA domain
- DINI-domain CLARA, over the DANRA domain (points inside DANRA's rectangle)
- DINI-domain CLARA, over the rest of its domain

Winds are compared as speeds (each grid has its own grid-relative winds).
Also maps of the DINI-domain ensemble-mean error and of a member at the last
lead time, with DANRA's outline.

Usage:
    uv run python dev-utils/dini_domain/compare_domains.py \\
        --danra-workdir <standard run> --dini-workdir <DINI-domain run> \\
        --danra-artifact artifact/build/clara-exact-stats --out-dir <dir>
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from clara_inference.convert_output import projection_crs

MEMBER = "ensemble_member"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a9a5b"
INK, INK_2 = "#1a1a1a", "#5c5c5c"
# (name, units, from single levels or a pressure level, conversion)
VARIABLES = [
    ("t2m", "K", None),
    ("pres_seasurface", "hPa", None),
    ("ws10m", "m/s", None),
    ("t850", "K", 850),
    ("ws850", "m/s", 850),
]


def open_fields(workdir: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Open the forecast and the DINI fields it is compared with.

    Parameters
    ----------
    workdir : Path
        A run's workdir (outputs/, inputs/).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Forecast (with ensemble_member) and DINI, with the VARIABLES, at the
        forecast's valid times.
    """

    def select(ds_sl: xr.Dataset, ds_pl: xr.Dataset) -> xr.Dataset:
        level = "pressure" if "pressure" in ds_pl.dims else "level"
        pl = ds_pl.sel({level: 850})
        return xr.Dataset(
            dict(
                t2m=ds_sl.t2m,
                pres_seasurface=ds_sl.pres_seasurface / 100.0,
                ws10m=np.hypot(ds_sl.u10m, ds_sl.v10m),
                t850=pl.t,
                ws850=np.hypot(pl.u, pl.v),
            )
        )

    out = workdir / "outputs"
    ds_fc = select(
        xr.open_zarr(out / "single_levels.zarr"),
        xr.open_zarr(out / "pressure_levels.zarr"),
    )
    inp = workdir / "inputs"
    ds_dini = select(
        xr.open_zarr(inp / "interior_single_levels.zarr"),
        xr.open_zarr(inp / "interior_pressure_levels.zarr"),
    ).sel(time=ds_fc.time)
    return ds_fc, ds_dini


def calculate_danra_mask(ds: xr.Dataset, danra_artifact: Path) -> xr.DataArray:
    """
    Mask the points of a grid that lie inside DANRA's rectangle.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with 2D lat/lon (y, x).
    danra_artifact : Path
        The DANRA-domain artifact (projection and grid).

    Returns
    -------
    xr.DataArray
        True inside DANRA, (y, x).
    """
    proj = projection_crs(
        danra_artifact / "configs" / "danra_model1_config_ifs_finetune.yaml"
    )
    grid = xr.open_zarr(
        danra_artifact / "grids" / "danra_model1_config_ifs_finetune.grid.zarr"
    )
    xy = proj.transform_points(
        ccrs.PlateCarree(), ds.lon.values, ds.lat.values
    )
    x, y = xy[..., 0], xy[..., 1]
    inside = (
        (x >= float(grid.x.min()))
        & (x <= float(grid.x.max()))
        & (y >= float(grid.y.min()))
        & (y <= float(grid.y.max()))
    )
    return xr.DataArray(inside, dims=("y", "x"))


def calculate_scores(
    ds_fc: xr.Dataset, ds_dini: xr.Dataset, mask: xr.DataArray | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the ensemble-mean RMSE and the spread per lead time.

    Parameters
    ----------
    ds_fc, ds_dini : xr.Dataset
        Forecast and DINI (open_fields).
    mask : xr.DataArray or None
        Points to include (all if None).

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Per variable: RMSE and spread per valid time.
    """
    scores = {}
    for v, _, _ in VARIABLES:
        fc, truth = ds_fc[v], ds_dini[v]
        err2 = (fc.mean(MEMBER) - truth) ** 2
        var = fc.var(MEMBER, ddof=1)
        if mask is not None:
            err2, var = err2.where(mask), var.where(mask)
        scores[v] = (
            np.sqrt(err2.mean(("y", "x"))).values,
            np.sqrt(var.mean(("y", "x"))).values,
        )
    return scores


def plot_scores(
    lead_h: np.ndarray,
    curves: list[tuple[str, str, dict]],
    fp: Path,
    title: str,
) -> None:
    """
    Plot RMSE (solid) and spread (dashed) per lead time, one panel per
    variable.

    Parameters
    ----------
    lead_h : np.ndarray
        Lead times (h).
    curves : list[tuple[str, str, dict]]
        (label, colour, calculate_scores output).
    fp : Path
        Output PNG.
    title : str
        Figure title.
    """
    fig, axes = plt.subplots(
        1, len(VARIABLES), figsize=(4 * len(VARIABLES), 3.6), sharex=True
    )
    for ax, (v, units, _) in zip(axes, VARIABLES):
        for label, colour, scores in curves:
            rmse, spread = scores[v]
            ax.plot(lead_h, rmse, color=colour, lw=2, label=label)
            ax.plot(lead_h, spread, color=colour, lw=1.5, ls="--")
        ax.set_title(f"{v} ({units})", color=INK)
        ax.set_xlabel("lead time (h)", color=INK_2)
        ax.set_ylim(bottom=0)
        ax.grid(color="#e6e6e6", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("RMSE (solid), spread (dashed)", color=INK_2)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(fp, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"wrote {fp}")


def plot_maps(
    ds_fc: xr.Dataset,
    ds_dini: xr.Dataset,
    danra_mask: xr.DataArray,
    fp: Path,
    title: str,
) -> None:
    """
    Map, at the last valid time: DINI, a CLARA member and the ensemble-mean
    error, of t2m and the 10 m wind speed, with DANRA's outline.

    Parameters
    ----------
    ds_fc, ds_dini : xr.Dataset
        DINI-domain forecast and DINI (open_fields).
    danra_mask : xr.DataArray
        Inside DANRA.
    fp : Path
        Output PNG.
    title : str
        Figure title.
    """
    t = ds_fc.time.values[-1]
    pc = ccrs.PlateCarree()
    proj = ccrs.LambertConformal(
        central_longitude=-8,
        central_latitude=55.5,
        standard_parallels=(55.5, 55.5),
    )
    rows = [
        ("t2m", "K", dict(cmap="RdYlBu_r", vmin=255, vmax=295), 6),
        ("ws10m", "m/s", dict(cmap="viridis", vmin=0, vmax=25), 8),
    ]
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9.5), subplot_kw=dict(projection=proj)
    )
    lat, lon = ds_fc.lat.values, ds_fc.lon.values
    for r, (v, units, style, err_max) in enumerate(rows):
        fields = [
            ("DINI", ds_dini[v].sel(time=t), style),
            ("CLARA member 1", ds_fc[v].sel(time=t).isel({MEMBER: 0}), style),
            (
                "CLARA ens. mean − DINI",
                ds_fc[v].sel(time=t).mean(MEMBER) - ds_dini[v].sel(time=t),
                dict(cmap="RdBu_r", vmin=-err_max, vmax=err_max),
            ),
        ]
        for c, (label, da, st) in enumerate(fields):
            ax = axes[r, c]
            m = ax.pcolormesh(
                lon,
                lat,
                da.values,
                transform=pc,
                shading="auto",
                rasterized=True,
                **st,
            )
            ax.contour(
                lon,
                lat,
                danra_mask.values.astype(float),
                levels=[0.5],
                colors="k",
                linewidths=1.2,
                transform=pc,
            )
            ax.add_feature(cfeature.COASTLINE, lw=0.4, color="#444444")
            ax.set_title(f"{label}: {v}", color=INK, fontsize=11)
            fig.colorbar(m, ax=ax, shrink=0.75, label=units)
    valid = str(t)[:13]
    fig.suptitle(f"{title}, valid {valid}Z (black: DANRA domain)", color=INK)
    fig.tight_layout()
    fig.savefig(fp, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {fp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--danra-workdir", required=True, type=Path)
    parser.add_argument("--dini-workdir", required=True, type=Path)
    parser.add_argument("--danra-artifact", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds_danra_fc, ds_danra_truth = open_fields(args.danra_workdir)
    ds_dini_fc, ds_dini_truth = open_fields(args.dini_workdir)
    times = np.intersect1d(ds_danra_fc.time.values, ds_dini_fc.time.values)
    ds_danra_fc, ds_danra_truth = (
        ds.sel(time=times).load() for ds in (ds_danra_fc, ds_danra_truth)
    )
    ds_dini_fc, ds_dini_truth = (
        ds.sel(time=times).load() for ds in (ds_dini_fc, ds_dini_truth)
    )
    mask = calculate_danra_mask(ds_dini_fc, args.danra_artifact)
    # lead times from the analysis time: the first DINI state of the inputs
    init = xr.open_zarr(
        args.dini_workdir / "inputs" / "interior_single_levels.zarr"
    ).time.values[0]
    lead_h = (times - init) / np.timedelta64(1, "h")
    curves = [
        (
            "CLARA (DANRA domain)",
            BLUE,
            calculate_scores(ds_danra_fc, ds_danra_truth, None),
        ),
        (
            "DINI-domain CLARA, inside DANRA",
            ORANGE,
            calculate_scores(ds_dini_fc, ds_dini_truth, mask),
        ),
        (
            "DINI-domain CLARA, outside DANRA",
            GREEN,
            calculate_scores(ds_dini_fc, ds_dini_truth, ~mask),
        ),
    ]
    title = (
        f"CLARA from {str(init)[:13]}Z against DINI, "
        f"{ds_dini_fc.sizes[MEMBER]} members"
    )
    plot_scores(lead_h, curves, args.out_dir / "dini_domain_scores.png", title)
    for label, _, scores in curves:
        print(label)
        for v, units, _ in VARIABLES:
            rmse, spread = scores[v]
            print(
                f"  {v:16s} RMSE {np.array2string(rmse, precision=2)}  "
                f"spread {np.array2string(spread, precision=2)} {units}"
            )
    plot_maps(
        ds_dini_fc,
        ds_dini_truth,
        mask,
        args.out_dir / "dini_domain_maps.png",
        "DINI-domain CLARA",
    )


if __name__ == "__main__":
    main()

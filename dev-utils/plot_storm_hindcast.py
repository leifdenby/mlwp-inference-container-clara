"""
Figures of a CLARA storm hindcast (`python -m clara_inference` with
`--initial-state-source danra`), compared with DANRA, from its workdir:

- `<prefix>_wind_mslp.png`: 10 m wind speed (0-40 m/s) with mean sea level
  pressure contours at one valid time: DANRA, the ensemble mean (mean of the
  members' wind speeds, mean pressure) and the first members
- `<prefix>_timeseries.png`: the minimum mean sea level pressure and the
  maximum 10 m wind speed over Denmark (land points of Denmark, Natural
  Earth 10m, without Bornholm's surroundings at sea) by valid time: DANRA,
  each member and the mean of the members, with observed values marked

DANRA is taken from the workdir (`inputs/interior_single_levels.zarr`, which
covers the initial states and all valid times). Winds are grid-relative in
DANRA and CLARA; the speed doesn't depend on that.

Usage (cartopy downloads Natural Earth data on first use), e.g. for the 3
December 1999 storm:

    uv run python dev-utils/plot_storm_hindcast.py \\
        --workdir /data/clara-hindcasts/1999-12-03_storm/init-1999-12-03T00 \\
        --valid-time 1999-12-03T18:00 --out-dir <dir> \\
        --observed-min-mslp 952 --observed-max-wind 41.2
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr

MEMBER = "ensemble_member"
# as in plot_example_forecast.py: categorical slots 1 and 2, text and surface
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d6d5d0"
# the DANRA projection (extra.projection in the interior datastore config)
DANRA_CRS = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=(56.7, 56.7),
    globe=ccrs.Globe(semimajor_axis=6367470.0, semiminor_axis=6367470.0),
)
# Denmark-centred map extent (lon_min, lon_max, lat_min, lat_max)
MAP_EXTENT = [3.5, 17.5, 53.3, 59.0]
WIND_STYLE = dict(cmap="YlGnBu", vmin=0.0, vmax=40.0)
MSLP_LEVELS = np.arange(940, 1041, 4)
FIGSIZE = (16, 9)
DPI = 150


def calculate_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate the 10 m wind speed.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with `u10m` and `v10m` (m/s).

    Returns
    -------
    xr.DataArray
        The wind speed (m/s).
    """
    return np.hypot(ds.u10m, ds.v10m)


def create_denmark_mask(ds_grid: xr.Dataset) -> xr.DataArray:
    """
    Create a mask of the grid points in Denmark (Natural Earth 10m).

    Parameters
    ----------
    ds_grid : xr.Dataset
        Dataset with 2D `lat` and `lon` coordinates.

    Returns
    -------
    xr.DataArray
        True for the points inside Denmark's land area.
    """
    reader = shpreader.Reader(
        shpreader.natural_earth("10m", "cultural", "admin_0_countries")
    )
    denmark = shapely.union_all(
        [
            r.geometry
            for r in reader.records()
            if r.attributes["ADMIN"] == "Denmark"
        ]
    )
    inside = shapely.contains_xy(
        denmark, ds_grid.lon.values, ds_grid.lat.values
    )
    return xr.DataArray(inside, dims=ds_grid.lat.dims)


def add_map_axes(fig: plt.Figure, rect: list[float]) -> plt.Axes:
    """
    Add a Denmark-centred map axes in the DANRA projection.

    Parameters
    ----------
    fig : plt.Figure
        The figure.
    rect : list[float]
        The axes position [left, bottom, width, height].

    Returns
    -------
    plt.Axes
        The map axes.
    """
    ax = fig.add_axes(rect, projection=DANRA_CRS, facecolor=SURFACE)
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"), edgecolor=INK, linewidth=0.5
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    return ax


def draw_wind_mslp(
    ax: plt.Axes, da_wind: xr.DataArray, da_mslp: xr.DataArray, title: str
):
    """
    Draw the wind speed with mean sea level pressure contours.

    Parameters
    ----------
    ax : plt.Axes
        Map axes in the DANRA projection.
    da_wind : xr.DataArray
        10 m wind speed (m/s), dims (y, x).
    da_mslp : xr.DataArray
        Mean sea level pressure (hPa), dims (y, x).
    title : str
        Panel title; the minimum pressure is added.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The wind speed mesh (for the colour bar).
    """
    mesh = ax.pcolormesh(
        da_wind.x.values,
        da_wind.y.values,
        da_wind.values,
        transform=DANRA_CRS,
        shading="auto",
        rasterized=True,
        **WIND_STYLE,
    )
    cs = ax.contour(
        da_mslp.x.values,
        da_mslp.y.values,
        da_mslp.values,
        levels=MSLP_LEVELS,
        colors=INK,
        linewidths=0.7,
        transform=DANRA_CRS,
    )
    ax.clabel(cs, fmt="%d", fontsize=8)
    # the pressure minimum within the map
    i_min = np.unravel_index(np.nanargmin(da_mslp.values), da_mslp.shape)
    ax.plot(
        da_mslp.x.values[i_min[1]],
        da_mslp.y.values[i_min[0]],
        marker="x",
        color=INK,
        markersize=8,
        transform=DANRA_CRS,
    )
    ax.set_title(
        f"{title}  (min {float(np.nanmin(da_mslp.values)):.0f} hPa)",
        color=INK,
        fontsize=13,
        loc="left",
    )
    return mesh


def crop_to_map(ds: xr.Dataset) -> xr.Dataset:
    """
    Crop a DANRA-grid dataset to the points within the map extent.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 2D `lat`/`lon` coordinates.

    Returns
    -------
    xr.Dataset
        The dataset on the smallest x/y box covering the map extent.
    """
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT
    inside = (
        (ds.lon >= lon_min - 2)
        & (ds.lon <= lon_max + 2)
        & (ds.lat >= lat_min - 1)
        & (ds.lat <= lat_max + 1)
    ).values
    j, i = np.nonzero(inside)
    return ds.isel(
        y=slice(j.min(), j.max() + 1), x=slice(i.min(), i.max() + 1)
    )


def describe_inputs(ds_clara: xr.Dataset) -> str:
    """
    Describe a CLARA forecast's input sources briefly.

    Parameters
    ----------
    ds_clara : xr.Dataset
        CLARA output, with the `initial_state_source` and `boundary_source`
        attributes (DINI and IFS if absent, as before they were recorded).

    Returns
    -------
    str
        E.g. "DANRA initial states, ERA5 boundary".
    """
    initial = ds_clara.attrs.get("initial_state_source", "DINI").split()[0]
    boundary = ds_clara.attrs.get("boundary_source", "IFS").split()[0]
    return f"{initial} initial states, {boundary} boundary"


def plot_wind_mslp(
    ds_clara: xr.Dataset,
    ds_danra: xr.Dataset,
    valid_time: np.datetime64,
    members: list[int],
    fp: Path,
    case_name: str,
) -> None:
    """
    Plot DANRA, the ensemble mean and members at one valid time (16:9).

    Parameters
    ----------
    ds_clara : xr.Dataset
        CLARA single levels (`outputs/single_levels.zarr`).
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    valid_time : np.datetime64
        The valid time.
    members : list[int]
        The members to show (up to 4).
    fp : Path
        The PNG to write.
    case_name : str
        The case, for the title.
    """
    ds_clara_t = crop_to_map(ds_clara.sel(time=valid_time)).load()
    ds_danra_t = crop_to_map(ds_danra.sel(time=valid_time)).load()
    analysis_time = np.datetime64(ds_clara.attrs["analysis_time"])
    # the forecast is initialised from the states at T and T + 3 h
    init_time = analysis_time + np.timedelta64(3, "h")
    lead_h = int((valid_time - init_time) / np.timedelta64(1, "h"))

    da_wind = calculate_wind_speed(ds_clara_t)
    da_mslp = ds_clara_t.pres_seasurface / 100
    panels = [
        (
            "DANRA",
            calculate_wind_speed(ds_danra_t),
            ds_danra_t.pres_seasurface / 100,
        ),
        ("CLARA ensemble mean", da_wind.mean(MEMBER), da_mslp.mean(MEMBER)),
    ] + [
        (
            f"CLARA member {m + 1}",
            da_wind.sel({MEMBER: m}),
            da_mslp.sel({MEMBER: m}),
        )
        for m in members
    ]

    fig = plt.figure(figsize=FIGSIZE, facecolor=SURFACE)
    n_cols = 3
    width, height = 0.31, 0.37
    for k, (title, da_w, da_p) in enumerate(panels[:6]):
        row, col = divmod(k, n_cols)
        ax = add_map_axes(
            fig, [0.02 + col * 0.325, 0.51 - row * 0.41, width, height]
        )
        mesh = draw_wind_mslp(ax, da_w, da_p, title)
    cb = fig.colorbar(
        mesh,
        cax=fig.add_axes([0.30, 0.065, 0.40, 0.018]),
        orientation="horizontal",
    )
    cb.set_label(
        "10 m wind speed (m/s); contours: MSLP every 4 hPa",
        color=INK_2,
        fontsize=11,
    )
    cb.ax.tick_params(colors=INK_2, labelsize=10)
    cb.outline.set_edgecolor(GRID)
    fig.text(
        0.02,
        0.975,
        f"{case_name}: valid {str(valid_time)[:13]}Z, CLARA "
        f"+{lead_h} h from {str(init_time)[:13]}Z "
        f"({ds_clara.sizes[MEMBER]} members; {describe_inputs(ds_clara)})",
        color=INK,
        fontsize=15,
        ha="left",
        va="top",
    )
    fig.savefig(fp, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {fp}")


def calculate_denmark_extremes(
    ds: xr.Dataset, da_mask: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate the minimum MSLP and maximum 10 m wind speed over Denmark.

    Parameters
    ----------
    ds : xr.Dataset
        Single levels with `pres_seasurface`, `u10m`, `v10m`.
    da_mask : xr.DataArray
        True for the points in Denmark, dims (y, x).

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Minimum MSLP (hPa) and maximum wind speed (m/s), over (y, x).
    """
    da_mslp = (ds.pres_seasurface / 100).where(da_mask)
    da_wind = calculate_wind_speed(ds).where(da_mask)
    return da_mslp.min(("y", "x")).load(), da_wind.max(("y", "x")).load()


def plot_timeseries(
    ds_clara: xr.Dataset,
    ds_danra: xr.Dataset,
    da_mask: xr.DataArray,
    fp: Path,
    case_name: str,
    observed_min_mslp: float | None = None,
    observed_max_wind: float | None = None,
    observed_time: np.datetime64 | None = None,
) -> None:
    """
    Plot the minimum MSLP and maximum wind over Denmark by valid time (16:9).

    Parameters
    ----------
    ds_clara : xr.Dataset
        CLARA single levels.
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    da_mask : xr.DataArray
        True for the points in Denmark.
    fp : Path
        The PNG to write.
    case_name : str
        The case, for the title.
    observed_min_mslp : float, optional
        Observed minimum pressure (hPa) to mark.
    observed_max_wind : float, optional
        Observed maximum mean wind (m/s) to mark.
    observed_time : np.datetime64, optional
        The time of the observed values.
    """
    da_mslp, da_wind = calculate_denmark_extremes(ds_clara, da_mask)
    # DANRA up to the last forecast time (the inputs have extra steps)
    da_mslp_danra, da_wind_danra = calculate_denmark_extremes(
        ds_danra.sel(time=slice(None, ds_clara.time.values[-1])), da_mask
    )
    analysis_time = np.datetime64(ds_clara.attrs["analysis_time"])
    init_time = analysis_time + np.timedelta64(3, "h")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=SURFACE)
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.83, bottom=0.12, wspace=0.18
    )
    panels = [
        (
            axes[0],
            da_mslp,
            da_mslp_danra,
            "Minimum MSLP over Denmark (hPa)",
            observed_min_mslp,
        ),
        (
            axes[1],
            da_wind,
            da_wind_danra,
            "Maximum 10 m wind speed over Denmark (m/s)",
            observed_max_wind,
        ),
    ]
    for ax, da_members, da_danra, title, observed in panels:
        ax.set_facecolor(SURFACE)
        for k, m in enumerate(da_members[MEMBER].values):
            ax.plot(
                da_members.time.values,
                da_members.sel({MEMBER: m}).values,
                color=SERIES_1,
                linewidth=1.0,
                alpha=0.6,
                label="CLARA members" if k == 0 else None,
            )
        ax.plot(
            da_members.time.values,
            da_members.mean(MEMBER).values,
            color=SERIES_2,
            linewidth=2.5,
            label="CLARA mean of members",
        )
        ax.plot(
            da_danra.time.values,
            da_danra.values,
            color=INK,
            linewidth=2.5,
            label="DANRA",
        )
        if observed is not None and observed_time is not None:
            ax.plot(
                observed_time,
                observed,
                marker="*",
                markersize=16,
                color=INK,
                markerfacecolor="white",
                linestyle="none",
                label=f"observed ({observed:g})",
            )
        ax.axvline(init_time, color=INK_2, linewidth=1.0, linestyle="--")
        ax.text(
            init_time,
            0.02,
            " forecast start",
            transform=ax.get_xaxis_transform(),
            color=INK_2,
            fontsize=11,
        )
        ax.set_title(title, color=INK, fontsize=15, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H UTC"))
        ax.tick_params(colors=INK_2, labelsize=12)
        ax.grid(color=GRID, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.legend(fontsize=12, frameon=False, labelcolor=INK)
    fig.text(
        0.06,
        0.96,
        f"{case_name}: CLARA from {str(init_time)[:13]}Z "
        f"({ds_clara.sizes[MEMBER]} members) and DANRA, land points of "
        "Denmark",
        color=INK,
        fontsize=15,
        ha="left",
        va="top",
    )
    fig.text(
        0.06,
        0.91,
        f"CLARA: {describe_inputs(ds_clara)}",
        color=INK_2,
        fontsize=12,
        ha="left",
        va="top",
    )
    fig.savefig(fp, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {fp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--valid-time", required=True, type=np.datetime64)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--prefix",
        default=None,
        help="file name prefix (default: clara_hindcast_<forecast start>)",
    )
    parser.add_argument("--members", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--case-name",
        default="3 December 1999 storm",
        help="the case, for the titles (default: %(default)s)",
    )
    parser.add_argument("--observed-min-mslp", type=float)
    parser.add_argument("--observed-max-wind", type=float)
    parser.add_argument(
        "--observed-time",
        type=np.datetime64,
        help="time of the observed values (default: --valid-time)",
    )
    args = parser.parse_args()

    ds_clara = xr.open_zarr(args.workdir / "outputs" / "single_levels.zarr")
    ds_danra = xr.open_zarr(
        args.workdir / "inputs" / "interior_single_levels.zarr"
    )
    valid_time = np.datetime64(args.valid_time, "ns")
    init_time = np.datetime64(
        ds_clara.attrs["analysis_time"]
    ) + np.timedelta64(3, "h")
    prefix = args.prefix or (
        f"clara_hindcast_{str(init_time)[:13].replace(':', '')}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_wind_mslp(
        ds_clara,
        ds_danra,
        valid_time,
        args.members,
        args.out_dir / f"{prefix}_wind_mslp.png",
        args.case_name,
    )
    plot_timeseries(
        ds_clara,
        ds_danra,
        create_denmark_mask(ds_danra),
        args.out_dir / f"{prefix}_timeseries.png",
        args.case_name,
        observed_min_mslp=args.observed_min_mslp,
        observed_max_wind=args.observed_max_wind,
        observed_time=(
            np.datetime64(args.observed_time or args.valid_time, "ns")
        ),
    )


if __name__ == "__main__":
    main()

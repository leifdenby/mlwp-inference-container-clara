"""
Slide figures of the net surface short-wave radiation (`swavr0m`) of CLARA
members against DANRA, from a hindcast workdir (`python -m clara_inference`
with `--initial-state-source danra`), recreating the style of the CRPS-LAM
paper's Fig. 20 (arXiv:2510.09484v3, plasma colours, 0-350 W/m2):

- `<prefix>_swavr_members.gif`: 2x2 grid (DANRA reanalysis, CLARA members
  1-3), one frame per valid time in daylight on the day of `--key-time`
  (the paper's 72 h frame, held longer), a large time stamp per frame
- `<prefix>_swavr_members_72h.png`: the `--key-time` frame (static fallback)
- `<prefix>_swavr_check.png`: 2x3 grid at `--key-time` (DANRA, members 1-3,
  ensemble mean and standard deviation), to compare with the paper figure

Sized for a ~3.6 x 2.6 in box on a 16:9 slide: 1600 x 1150 px, white
background, no title, fonts of at least ~4% of the figure height. A few grid
points at the domain edge are trimmed so no boundary effects show.

Usage, e.g. for the paper's case (72 h valid 2019-11-01T12Z):

    uv run python dev-utils/plot_swavr_members.py \\
        --workdir <hindcast workdir> --key-time 2019-11-01T12:00 \\
        --out-dir <dir>
"""
import argparse
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from PIL import Image
from plot_storm_hindcast import DANRA_CRS, MEMBER

STYLE = dict(cmap="plasma", vmin=0.0, vmax=350.0)
STD_STYLE = dict(cmap="plasma", vmin=0.0, vmax=110.0)
WHITE = "#ffffff"
INK = "#0b0b0b"
COAST = "#e8e8e8"
FIG_W, FIG_H, DPI = 8.0, 5.75, 200  # 1600 x 1150 px
FONT = 17  # pt: ~4% of the figure height
EDGE_TRIM = 12  # grid points trimmed at each domain edge
# daylight frames: DANRA's 99th percentile of the flux above this (W/m2)
DAYLIGHT_MIN = 20.0
FRAME_MS = 800
KEY_FRAME_MS = 2500


def join_members(datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Join the members of several runs of the same forecast.

    Parameters
    ----------
    datasets : list[xr.Dataset]
        CLARA single levels of each run, with the same analysis time and
        valid times.

    Returns
    -------
    xr.Dataset
        All members, numbered from 0, with the first run's attributes.
    """
    analysis_times = {ds.attrs["analysis_time"] for ds in datasets}
    if len(analysis_times) != 1:
        raise ValueError(f"different analysis times: {analysis_times}")
    ds = xr.concat(datasets, dim=MEMBER, combine_attrs="override")
    ds = ds.assign_coords({MEMBER: np.arange(ds.sizes[MEMBER])})
    ds.attrs["ensemble_size"] = ds.sizes[MEMBER]
    return ds


def trim_edges(da: xr.DataArray, n: int = EDGE_TRIM) -> xr.DataArray:
    """
    Trim `n` grid points at each edge of the domain.

    Parameters
    ----------
    da : xr.DataArray
        Field with dims `x` and `y`.
    n : int
        Number of points to trim.

    Returns
    -------
    xr.DataArray
        The field without the edge points.
    """
    return da.isel(x=slice(n, -n), y=slice(n, -n))


def add_panel(
    fig: plt.Figure,
    rect: list[float],
    da: xr.DataArray,
    label: str,
    style: dict,
    label_at_bottom: bool = False,
):
    """
    Add a map panel of a field on the DANRA grid, labelled inside.

    Parameters
    ----------
    fig : plt.Figure
        The figure.
    rect : list[float]
        The panel position [left, bottom, width, height] (figure fraction).
    da : xr.DataArray
        The field, dims (y, x) in DANRA projection coordinates.
    label : str
        The panel label.
    style : dict
        pcolormesh colour arguments.
    label_at_bottom : bool, optional
        Put the label at the bottom left instead of the top left.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh (for a colour bar).
    """
    ax = fig.add_axes(rect, projection=DANRA_CRS)
    x, y = da.x.values, da.y.values
    ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=DANRA_CRS)
    mesh = ax.pcolormesh(
        x,
        y,
        da.transpose("y", "x").values,
        transform=DANRA_CRS,
        shading="auto",
        rasterized=True,
        **style,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor=COAST,
        linewidth=0.35,
        alpha=0.6,
    )
    ax.spines["geo"].set_visible(False)
    ax.text(
        0.02,
        0.03 if label_at_bottom else 0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom" if label_at_bottom else "top",
        fontsize=FONT,
        color=WHITE,
        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.45),
    )
    return mesh


def add_colorbar(
    fig: plt.Figure,
    rect: list[float],
    style: dict,
    label: str = "net shortwave radiation at the surface (W/m²)",
) -> None:
    """
    Add a slim horizontal colour bar (by default of the short-wave flux).

    Parameters
    ----------
    fig : plt.Figure
        The figure.
    rect : list[float]
        The colour bar position [left, bottom, width, height].
    style : dict
        The colour arguments of the panels.
    label : str, optional
        The colour bar label.
    """
    sm = plt.cm.ScalarMappable(
        norm=Normalize(style["vmin"], style["vmax"]), cmap=style["cmap"]
    )
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_label(label, fontsize=FONT, color=INK)
    cb.ax.tick_params(labelsize=FONT, colors=INK, length=3)
    cb.outline.set_visible(False)


def render_members_frame(
    panels: list[tuple[str, xr.DataArray]], time_stamp: str
) -> plt.Figure:
    """
    Render the 2x2 frame: four panels, colour bar and time stamp.

    Parameters
    ----------
    panels : list[tuple[str, xr.DataArray]]
        Four (label, field) pairs.
    time_stamp : str
        E.g. "12 UTC\\n+72 h", shown large to the right of the grid.

    Returns
    -------
    plt.Figure
        The figure (1600 x 1150 px).
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=WHITE)
    da0 = panels[0][1]
    aspect = np.ptp(da0.x.values) / np.ptp(da0.y.values)
    # grid at the top left, the colour bar under it, the time stamp right
    cb_h_in, gap_in = 0.95, 0.04
    panel_h_in = (FIG_H - cb_h_in - gap_in) / 2
    panel_w_in = panel_h_in * aspect
    w, h = panel_w_in / FIG_W, panel_h_in / FIG_H
    gx, gy = gap_in / FIG_W, gap_in / FIG_H
    for k, (label, da) in enumerate(panels):
        row, col = divmod(k, 2)
        add_panel(
            fig,
            [col * (w + gx), 1 - (row + 1) * h - row * gy, w, h],
            da,
            label,
            STYLE,
        )
    grid_w = 2 * w + gx
    add_colorbar(fig, [0.03, 0.115, grid_w - 0.06, 0.022], STYLE)
    fig.text(
        (grid_w + 1) / 2,
        1 - h,
        time_stamp,
        ha="center",
        va="center",
        fontsize=1.5 * FONT,
        color=INK,
        linespacing=1.3,
    )
    return fig


def convert_figure_to_image(fig: plt.Figure) -> Image.Image:
    """
    Convert a figure to an RGB image (and close it).

    Parameters
    ----------
    fig : plt.Figure
        The figure.

    Returns
    -------
    Image.Image
        The rendered figure.
    """
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert(
        "RGB"
    )
    plt.close(fig)
    return image


def save_gif(
    frames: list[Image.Image],
    durations: list[int],
    fp: Path,
    colors: int = 256,
) -> None:
    """
    Save frames as a looping GIF with one shared colour palette.

    Parameters
    ----------
    frames : list[Image.Image]
        The frames (RGB).
    durations : list[int]
        Display time of each frame (ms).
    fp : Path
        The GIF to write.
    colors : int, optional
        Palette size (fewer colours compress better).
    """
    thumbs = [f.reduce(4) for f in frames]
    sheet = Image.new("RGB", (thumbs[0].width, thumbs[0].height * len(thumbs)))
    for k, thumb in enumerate(thumbs):
        sheet.paste(thumb, (0, k * thumb.height))
    palette = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames_p = [
        f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames
    ]
    frames_p[0].save(
        fp,
        save_all=True,
        append_images=frames_p[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=1,
    )


def select_daylight_times(
    da_danra: xr.DataArray, key_time: np.datetime64
) -> np.ndarray:
    """
    Select the valid times in daylight on the day of `key_time`.

    Parameters
    ----------
    da_danra : xr.DataArray
        DANRA's short-wave flux at the forecast's valid times.
    key_time : np.datetime64
        A time of the day to show.

    Returns
    -------
    np.ndarray
        The valid times with DANRA's 99th percentile above DAYLIGHT_MIN.
    """
    day = key_time.astype("datetime64[D]")
    da_day = da_danra.sel(time=da_danra.time.dt.floor("D") == day)
    q99 = da_day.quantile(0.99, dim=("x", "y"))
    return da_day.time.values[(q99 > DAYLIGHT_MIN).values]


def plot_check(
    da_danra: xr.DataArray,
    da_clara: xr.DataArray,
    members: list[int],
    time_stamp: str,
    fp: Path,
) -> None:
    """
    Plot DANRA, three members, ensemble mean and std (2x3) at one time.

    Parameters
    ----------
    da_danra : xr.DataArray
        DANRA's flux at the time.
    da_clara : xr.DataArray
        CLARA's flux at the time, with the member dimension.
    members : list[int]
        The three members to show (0-based).
    time_stamp : str
        The valid time and lead, for the title.
    fp : Path
        The PNG to write.
    """
    panels = (
        [("DANRA reanalysis", da_danra, STYLE)]
        + [
            (f"CLARA member {m + 1}", da_clara.sel({MEMBER: m}), STYLE)
            for m in members
        ]
        + [
            ("CLARA ens. mean", da_clara.mean(MEMBER), STYLE),
            ("CLARA ens. std", da_clara.std(MEMBER), STD_STYLE),
        ]
    )
    fig = plt.figure(figsize=(15, 8.4), dpi=110, facecolor=WHITE)
    w, h = 0.31, 0.39
    for k, (label, da, style) in enumerate(panels):
        row, col = divmod(k, 3)
        mesh = add_panel(
            fig,
            [0.02 + col * 0.325, 0.56 - row * 0.43, w, h],
            da,
            label,
            style,
        )
        if label.endswith("std"):
            cb = fig.colorbar(
                mesh,
                cax=fig.add_axes([0.67, 0.075, 0.31, 0.015]),
                orientation="horizontal",
            )
            cb.set_label("ens. std (W/m²)", fontsize=12)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=Normalize(STYLE["vmin"], STYLE["vmax"]), cmap=STYLE["cmap"]
        ),
        cax=fig.add_axes([0.02, 0.075, 0.635, 0.015]),
        orientation="horizontal",
    )
    cb.set_label("net shortwave radiation at the surface (W/m²)", fontsize=12)
    fig.text(
        0.02,
        0.985,
        f"CLARA vs DANRA, swavr0m, {time_stamp} "
        f"({da_clara.sizes[MEMBER]} members)",
        fontsize=14,
        va="top",
    )
    fig.savefig(fp, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {fp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workdir",
        required=True,
        type=Path,
        nargs="+",
        help="hindcast workdir(s); the members of several runs of the same "
        "forecast (e.g. split for memory) are joined",
    )
    parser.add_argument("--key-time", required=True, type=np.datetime64)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--prefix",
        default=None,
        help="file name prefix (default: clara_<forecast start>)",
    )
    parser.add_argument(
        "--members",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="the three members to show, 0-based (default: %(default)s)",
    )
    args = parser.parse_args()

    ds_clara = join_members(
        [
            xr.open_zarr(w / "outputs" / "single_levels.zarr")
            for w in args.workdir
        ]
    )
    ds_danra = xr.open_zarr(
        args.workdir[0] / "inputs" / "interior_single_levels.zarr"
    )
    da_clara = trim_edges(ds_clara.swavr0m).load()
    da_danra = trim_edges(ds_danra.swavr0m.sel(time=ds_clara.time)).load()
    # the forecast is initialised from the states at T and T + 3 h
    init_time = np.datetime64(
        ds_clara.attrs["analysis_time"]
    ) + np.timedelta64(3, "h")
    key_time = np.datetime64(args.key_time, "ns")
    prefix = args.prefix or f"clara_{str(init_time)[:13]}Z"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def format_time_stamp(t: np.datetime64, sep: str) -> str:
        lead_h = int((t - init_time) / np.timedelta64(1, "h"))
        return f"{str(t)[11:13]} UTC{sep}+{lead_h} h"

    frames, durations = [], []
    for t in select_daylight_times(da_danra, key_time):
        panels = [("DANRA reanalysis", da_danra.sel(time=t))] + [
            (f"CLARA member {m + 1}", da_clara.sel(time=t, **{MEMBER: m}))
            for m in args.members
        ]
        fig = render_members_frame(panels, format_time_stamp(t, "\n"))
        if t == key_time:
            fp_png = args.out_dir / f"{prefix}_swavr_members_72h.png"
            fig.savefig(fp_png, dpi=DPI, facecolor=WHITE)
            print(f"wrote {fp_png}")
        frames.append(convert_figure_to_image(fig))
        durations.append(KEY_FRAME_MS if t == key_time else FRAME_MS)
        print(f"frame {str(t)[:13]}")
    fp_gif = args.out_dir / f"{prefix}_swavr_members.gif"
    save_gif(frames, durations, fp_gif)
    print(f"wrote {fp_gif} ({len(frames)} frames, {durations} ms)")

    plot_check(
        da_danra.sel(time=key_time),
        da_clara.sel(time=key_time),
        args.members,
        f"valid {str(key_time)[:13]}Z, "
        f"{format_time_stamp(key_time, ' ')} from {str(init_time)[:13]}Z",
        args.out_dir / f"{prefix}_swavr_check.png",
    )


if __name__ == "__main__":
    main()

"""
Slide animation of the net surface long-wave radiation loss of CLARA members
against DANRA, from a hindcast workdir: the companion of
`plot_swavr_members.py` (same case, members, size and style), for the cloud
cover at all hours.

`lwavr0m` is the net long-wave flux at the surface, negative for a net loss
(about -130 W/m2 under a clear sky, close to 0 under thick low cloud; there
is no top-of-atmosphere outgoing long-wave field). Shown is the loss,
-lwavr0m, in blues: cloud light (small loss), clear sky dark (large loss),
with a fixed colour range, the 2nd-98th percentiles of DANRA over all frames.

- `<prefix>_lwavr_members.gif`: 2x2 grid (DANRA reanalysis, CLARA members
  1-3), one frame per 3-hourly valid time of the forecast, the `--key-time`
  frame held longer, the time stamp in the top left of the DANRA panel
- `<prefix>_lwavr_members_72h.png`: the `--key-time` frame

The grid fills the frame (1600 x 1150 px), so the domain is cropped a little
north and south (besides the edge trim) to the panels' aspect.

Usage, e.g. for the paper's case (72 h valid 2019-11-01T12Z):

    uv run python dev-utils/plot_lwavr_members.py \\
        --workdir <hindcast workdir(s)> --key-time 2019-11-01T12:00 \\
        --out-dir <dir>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from PIL import Image
from plot_storm_hindcast import MEMBER
from plot_swavr_members import (
    DPI,
    FIG_H,
    FIG_W,
    FONT,
    FRAME_MS,
    KEY_FRAME_MS,
    WHITE,
    add_colorbar,
    add_panel,
    convert_figure_to_image,
    join_members,
    save_gif,
    trim_edges,
)

# blues from a very light blue (not white, so that cloudy panel edges stay
# visible against the white background)
CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.12, 1.0, 256)))
PERCENTILES = (2, 98)
CB_LABEL = "net longwave radiation loss at the surface (W/m²)"
# layout (inches): the colour bar strip under the grid, the gap between
# panels
CB_STRIP_IN, GAP_IN = 0.95, 0.04
# GIF palette size and frame width: 25 frames of smooth fields at 1600 px
# with 256 colours are ~20 MB; 1080 px is still ~300 px per inch in the
# slide's ~3.6 in box (the PNG keeps 1600 px)
GIF_COLORS = 32
GIF_WIDTH = 1080


def calculate_panel_aspect() -> float:
    """
    Calculate the panel aspect (width / height) of a grid filling the frame.

    Returns
    -------
    float
        The aspect of each of the 2x2 panels.
    """
    panel_w = (FIG_W - GAP_IN) / 2
    panel_h = (FIG_H - CB_STRIP_IN - GAP_IN) / 2
    return panel_w / panel_h


def crop_to_aspect(da: xr.DataArray, aspect: float) -> xr.DataArray:
    """
    Crop the domain symmetrically in y (or x) to an aspect.

    Parameters
    ----------
    da : xr.DataArray
        Field with dims `x` and `y` (equally spaced).
    aspect : float
        The width / height to crop to.

    Returns
    -------
    xr.DataArray
        The cropped field.
    """
    nx, ny = da.sizes["x"], da.sizes["y"]
    if nx / ny < aspect:
        n = int(round(nx / aspect))
        j0 = (ny - n) // 2
        return da.isel(y=slice(j0, j0 + n))
    n = int(round(ny * aspect))
    i0 = (nx - n) // 2
    return da.isel(x=slice(i0, i0 + n))


def render_frame(
    panels: list[tuple[str, xr.DataArray]], time_stamp: str, style: dict
) -> plt.Figure:
    """
    Render the 2x2 frame filling the figure, with the colour bar under it.

    Parameters
    ----------
    panels : list[tuple[str, xr.DataArray]]
        Four (label, field) pairs; the first is DANRA.
    time_stamp : str
        E.g. "1 Nov 12 UTC, +72 h", in the top left of the first panel.
    style : dict
        pcolormesh colour arguments.

    Returns
    -------
    plt.Figure
        The figure (1600 x 1150 px).
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=WHITE)
    w = (1 - GAP_IN / FIG_W) / 2
    h = (1 - (CB_STRIP_IN + GAP_IN) / FIG_H) / 2
    gx, gy = GAP_IN / FIG_W, GAP_IN / FIG_H
    for k, (label, da) in enumerate(panels):
        row, col = divmod(k, 2)
        add_panel(
            fig,
            [col * (w + gx), 1 - (row + 1) * h - row * gy, w, h],
            da,
            label,
            style,
            label_at_bottom=True,
        )
    fig.text(
        0.008,
        1 - 0.012,
        time_stamp,
        ha="left",
        va="top",
        fontsize=1.3 * FONT,
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec="none", alpha=0.85),
    )
    add_colorbar(fig, [0.06, 0.125, 0.88, 0.022], style, label=CB_LABEL)
    return fig


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
    aspect = calculate_panel_aspect()
    # the loss: positive, large under a clear sky
    da_clara = -crop_to_aspect(trim_edges(ds_clara.lwavr0m), aspect).load()
    da_danra = -crop_to_aspect(
        trim_edges(ds_danra.lwavr0m.sel(time=ds_clara.time)), aspect
    ).load()
    vmin, vmax = np.nanpercentile(da_danra.values, PERCENTILES)
    style = dict(cmap=CMAP, vmin=float(vmin), vmax=float(vmax))
    print(f"colour range {vmin:.0f} .. {vmax:.0f} W/m2 ({PERCENTILES} %)")

    # the forecast is initialised from the states at T and T + 3 h
    init_time = np.datetime64(
        ds_clara.attrs["analysis_time"]
    ) + np.timedelta64(3, "h")
    key_time = np.datetime64(args.key_time, "ns")
    prefix = args.prefix or f"clara_{str(init_time)[:13]}Z"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames, durations = [], []
    for t in da_clara.time.values:
        lead_h = int((t - init_time) / np.timedelta64(1, "h"))
        valid = t.astype("datetime64[s]").item()
        time_stamp = f"{valid:%-d %b %H} UTC, +{lead_h} h"
        panels = [("DANRA reanalysis", da_danra.sel(time=t))] + [
            (f"CLARA member {m + 1}", da_clara.sel(time=t, **{MEMBER: m}))
            for m in args.members
        ]
        fig = render_frame(panels, time_stamp, style)
        if t == key_time:
            fp_png = args.out_dir / f"{prefix}_lwavr_members_72h.png"
            fig.savefig(fp_png, dpi=DPI, facecolor=WHITE)
            print(f"wrote {fp_png}")
        frames.append(convert_figure_to_image(fig))
        durations.append(KEY_FRAME_MS if t == key_time else FRAME_MS)
        print(f"frame {time_stamp}")
    fp_gif = args.out_dir / f"{prefix}_lwavr_members.gif"
    size = (GIF_WIDTH, round(GIF_WIDTH * FIG_H / FIG_W))
    frames = [f.resize(size, Image.Resampling.LANCZOS) for f in frames]
    save_gif(frames, durations, fp_gif, colors=GIF_COLORS)
    print(f"wrote {fp_gif} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

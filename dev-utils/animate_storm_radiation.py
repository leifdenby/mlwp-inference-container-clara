"""
Animation (GIF) of the cloud cover in a CLARA storm hindcast, compared with
DANRA, from its workdir (as for `plot_storm_hindcast.py`).

The cloud cover is shown by the net surface long-wave radiation flux
(`lwavr0m`, instantaneous): about -130 W/m2 under a clear sky, close to 0
under thick low cloud. It is drawn in greys with cloud white, like a
satellite image. (The short-wave flux would only show the few daylight hours
of a winter case.) Panels in a 2x2 grid: DANRA, the ensemble mean and two
members, one frame per valid time of the forecast. The ensemble mean is
smoother than any member, as for all ensemble means.

Usage, e.g. for the 3 December 1999 storm:

    uv run python dev-utils/animate_storm_radiation.py \\
        --workdir /data/clara-hindcasts/1999-12-03_storm/init-1999-12-03T00 \\
        --out-dir <dir>
"""
import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image
from plot_storm_hindcast import (
    DANRA_CRS,
    GRID,
    INK,
    INK_2,
    MEMBER,
    SURFACE,
    add_map_axes,
    crop_to_map,
    describe_inputs,
)

LW_STYLE = dict(cmap="Greys_r", vmin=-130.0, vmax=0.0)
FIGSIZE = (16, 9)
DPI = 100
FRAME_MS = 800
LAST_FRAME_MS = 2400


def draw_longwave(ax: plt.Axes, da_lw: xr.DataArray, title: str):
    """
    Draw the net surface long-wave radiation flux.

    Parameters
    ----------
    ax : plt.Axes
        Map axes in the DANRA projection.
    da_lw : xr.DataArray
        Net surface long-wave radiation flux (W/m2), dims (y, x).
    title : str
        Panel title.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh (for the colour bar).
    """
    mesh = ax.pcolormesh(
        da_lw.x.values,
        da_lw.y.values,
        da_lw.transpose("y", "x").values,
        transform=DANRA_CRS,
        shading="auto",
        rasterized=True,
        **LW_STYLE,
    )
    ax.set_title(title, color=INK, fontsize=14, loc="left")
    return mesh


def render_frame(
    panels: list[tuple[str, xr.DataArray]],
    title: str,
    full_domain: bool,
) -> Image.Image:
    """
    Render one frame of the 2x2 grid.

    Parameters
    ----------
    panels : list[tuple[str, xr.DataArray]]
        Four (title, long-wave flux) pairs.
    title : str
        The frame's title.
    full_domain : bool
        Show the whole DANRA domain instead of the Denmark-centred map.

    Returns
    -------
    Image.Image
        The frame (RGB).
    """
    fig = plt.figure(figsize=FIGSIZE, facecolor=SURFACE)
    # the maps are about as wide as tall, so the panels are centred with
    # the colour bar next to them
    width, height = 0.30, 0.40
    for k, (panel_title, da_lw) in enumerate(panels):
        row, col = divmod(k, 2)
        ax = add_map_axes(
            fig, [0.17 + col * 0.31, 0.475 - row * 0.445, width, height]
        )
        if full_domain:
            ax.set_extent(
                [
                    float(da_lw.x.min()),
                    float(da_lw.x.max()),
                    float(da_lw.y.min()),
                    float(da_lw.y.max()),
                ],
                crs=DANRA_CRS,
            )
        mesh = draw_longwave(ax, da_lw, panel_title)
    cb = fig.colorbar(
        mesh,
        cax=fig.add_axes([0.80, 0.12, 0.012, 0.70]),
        orientation="vertical",
    )
    cb.set_label(
        "net surface long-wave radiation (W/m²)\n"
        "white: cloudy, dark: clear sky",
        color=INK_2,
        fontsize=11,
    )
    cb.ax.tick_params(colors=INK_2, labelsize=10)
    cb.outline.set_edgecolor(GRID)
    fig.text(0.5, 0.975, title, color=INK, fontsize=15, ha="center", va="top")
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, facecolor=SURFACE, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def animate_longwave(
    ds_clara: xr.Dataset,
    ds_danra: xr.Dataset,
    members: list[int],
    fp: Path,
    case_name: str,
    full_domain: bool = False,
) -> None:
    """
    Animate the cloud cover (net long-wave flux) for all forecast times.

    Parameters
    ----------
    ds_clara : xr.Dataset
        CLARA single levels (`outputs/single_levels.zarr`).
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    members : list[int]
        The two members to show (0-based).
    fp : Path
        The GIF to write.
    case_name : str
        The case, for the title.
    full_domain : bool, optional
        Show the whole DANRA domain instead of the Denmark-centred map.
    """
    if len(members) != 2:
        raise ValueError(f"expected two members, got {members}")
    analysis_time = np.datetime64(ds_clara.attrs["analysis_time"])
    # the forecast is initialised from the states at T and T + 3 h
    init_time = analysis_time + np.timedelta64(3, "h")
    da_clara = ds_clara.lwavr0m
    da_danra = ds_danra.lwavr0m.sel(time=ds_clara.time)
    if not full_domain:
        da_clara, da_danra = crop_to_map(da_clara), crop_to_map(da_danra)
    da_clara, da_danra = da_clara.load(), da_danra.load()

    frames = []
    for t in da_clara.time.values:
        lead_h = int((t - init_time) / np.timedelta64(1, "h"))
        da_t = da_clara.sel(time=t)
        panels = [
            ("DANRA", da_danra.sel(time=t)),
            ("CLARA ensemble mean", da_t.mean(MEMBER)),
        ] + [(f"CLARA member {m + 1}", da_t.sel({MEMBER: m})) for m in members]
        title = (
            f"{case_name}: valid {str(t)[:13]}Z, CLARA +{lead_h} h from "
            f"{str(init_time)[:13]}Z ({ds_clara.sizes[MEMBER]} members; "
            f"{describe_inputs(ds_clara)})"
        )
        frames.append(render_frame(panels, title, full_domain))
        print(f"frame {str(t)[:13]}")
    durations = [FRAME_MS] * (len(frames) - 1) + [LAST_FRAME_MS]
    frames[0].save(
        fp,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {fp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--prefix",
        default=None,
        help="file name prefix (default: clara_hindcast_<forecast start>)",
    )
    parser.add_argument(
        "--members",
        type=int,
        nargs=2,
        default=[0, 1],
        help="the two members to show, 0-based (default: %(default)s)",
    )
    parser.add_argument(
        "--case-name",
        default="3 December 1999 storm",
        help="the case, for the titles (default: %(default)s)",
    )
    parser.add_argument(
        "--full-domain",
        action="store_true",
        help="show the whole DANRA domain (default: Denmark-centred)",
    )
    args = parser.parse_args()

    ds_clara = xr.open_zarr(args.workdir / "outputs" / "single_levels.zarr")
    ds_danra = xr.open_zarr(
        args.workdir / "inputs" / "interior_single_levels.zarr"
    )
    init_time = np.datetime64(
        ds_clara.attrs["analysis_time"]
    ) + np.timedelta64(3, "h")
    prefix = args.prefix or (
        f"clara_hindcast_{str(init_time)[:13].replace(':', '')}"
    )
    suffix = "_full_domain" if args.full_domain else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    animate_longwave(
        ds_clara,
        ds_danra,
        args.members,
        args.out_dir / f"{prefix}_clouds{suffix}.gif",
        args.case_name,
        full_domain=args.full_domain,
    )


if __name__ == "__main__":
    main()

"""
Animation (GIF) of the surface radiation (cloud cover) in a CLARA hindcast,
compared with DANRA, from its workdir (as for `plot_storm_hindcast.py`).

- `--variable lwavr0m` (default, `<prefix>_clouds.gif`): the net surface
  long-wave radiation flux (instantaneous), about -130 W/m2 under a clear
  sky and close to 0 under thick low cloud, so it shows the cloud cover at
  all hours. Drawn in greys with cloud white, like a satellite image.
- `--variable swavr0m` (`<prefix>_shortwave.gif`): the net surface
  short-wave radiation flux, in the colours of the paper's Fig. 20 (plasma,
  0-400 W/m2). Dark under cloud, and at night.

Panels in a 2x2 grid: DANRA, the ensemble mean and two members, one frame
per valid time of the forecast. The ensemble mean is smoother than any
member, as for all ensemble means. All frames share one colour palette (the
most widely supported GIF form).

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

VARIABLES = {
    "lwavr0m": dict(
        name="clouds",
        style=dict(cmap="Greys_r", vmin=-130.0, vmax=0.0),
        label="net surface long-wave radiation (W/m²)\n"
        "white: cloudy, dark: clear sky",
    ),
    "swavr0m": dict(
        name="shortwave",
        style=dict(cmap="plasma", vmin=0.0, vmax=400.0),
        label="net surface short-wave radiation (W/m²)\n"
        "dark: cloudy (or night)",
    ),
}
FIGSIZE = (16, 9)
DPI = 100
FRAME_MS = 800
LAST_FRAME_MS = 2400


def draw_radiation(ax: plt.Axes, da: xr.DataArray, title: str, style: dict):
    """
    Draw a radiation flux field.

    Parameters
    ----------
    ax : plt.Axes
        Map axes in the DANRA projection.
    da : xr.DataArray
        The flux (W/m2), dims (y, x).
    title : str
        Panel title.
    style : dict
        pcolormesh colour arguments (cmap, vmin, vmax).

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh (for the colour bar).
    """
    mesh = ax.pcolormesh(
        da.x.values,
        da.y.values,
        da.transpose("y", "x").values,
        transform=DANRA_CRS,
        shading="auto",
        rasterized=True,
        **style,
    )
    ax.set_title(title, color=INK, fontsize=14, loc="left")
    return mesh


def render_frame(
    panels: list[tuple[str, xr.DataArray]],
    title: str,
    variable: str,
    full_domain: bool,
) -> Image.Image:
    """
    Render one frame of the 2x2 grid.

    Parameters
    ----------
    panels : list[tuple[str, xr.DataArray]]
        Four (title, flux) pairs.
    title : str
        The frame's title.
    variable : str
        The variable (a key of VARIABLES), for the colours and label.
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
    for k, (panel_title, da) in enumerate(panels):
        row, col = divmod(k, 2)
        ax = add_map_axes(
            fig, [0.17 + col * 0.31, 0.475 - row * 0.445, width, height]
        )
        if full_domain:
            ax.set_extent(
                [
                    float(da.x.min()),
                    float(da.x.max()),
                    float(da.y.min()),
                    float(da.y.max()),
                ],
                crs=DANRA_CRS,
            )
        mesh = draw_radiation(
            ax, da, panel_title, VARIABLES[variable]["style"]
        )
    cb = fig.colorbar(
        mesh,
        cax=fig.add_axes([0.80, 0.12, 0.012, 0.70]),
        orientation="vertical",
    )
    cb.set_label(VARIABLES[variable]["label"], color=INK_2, fontsize=11)
    cb.ax.tick_params(colors=INK_2, labelsize=10)
    cb.outline.set_edgecolor(GRID)
    fig.text(0.5, 0.975, title, color=INK, fontsize=15, ha="center", va="top")
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, facecolor=SURFACE, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_gif(frames: list[Image.Image], fp: Path) -> None:
    """
    Save frames as a looping GIF with one shared colour palette.

    Parameters
    ----------
    frames : list[Image.Image]
        The frames (RGB).
    fp : Path
        The GIF to write.
    """
    # the palette from all frames side by side (subsampled), so that every
    # frame's colours are in it; without dithering for clean fields
    thumbs = [f.reduce(4) for f in frames]
    sheet = Image.new("RGB", (thumbs[0].width, thumbs[0].height * len(thumbs)))
    for k, thumb in enumerate(thumbs):
        sheet.paste(thumb, (0, k * thumb.height))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames_p = [
        f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames
    ]
    durations = [FRAME_MS] * (len(frames_p) - 1) + [LAST_FRAME_MS]
    frames_p[0].save(
        fp,
        save_all=True,
        append_images=frames_p[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=1,
    )


def animate_radiation(
    ds_clara: xr.Dataset,
    ds_danra: xr.Dataset,
    variable: str,
    members: list[int],
    fp: Path,
    case_name: str,
    full_domain: bool = False,
) -> None:
    """
    Animate a radiation flux (cloud cover) for all forecast times.

    Parameters
    ----------
    ds_clara : xr.Dataset
        CLARA single levels (`outputs/single_levels.zarr`).
    ds_danra : xr.Dataset
        DANRA single levels on the same grid.
    variable : str
        `lwavr0m` or `swavr0m` (a key of VARIABLES).
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
    da_clara = ds_clara[variable]
    da_danra = ds_danra[variable].sel(time=ds_clara.time)
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
        frames.append(render_frame(panels, title, variable, full_domain))
        print(f"frame {str(t)[:13]}")
    save_gif(frames, fp)
    print(f"wrote {fp} ({len(frames)} frames)")


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
    parser.add_argument(
        "--variable",
        choices=list(VARIABLES),
        default="lwavr0m",
        help="the radiation flux (default: %(default)s)",
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
    name = VARIABLES[args.variable]["name"]
    animate_radiation(
        ds_clara,
        ds_danra,
        args.variable,
        args.members,
        args.out_dir / f"{prefix}_{name}{suffix}.gif",
        args.case_name,
        full_domain=args.full_domain,
    )


if __name__ == "__main__":
    main()

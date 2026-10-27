"""
Map of the CLARA domain: the DANRA interior, the IFS boundary points the
model was trained with (400 km margin around DANRA, interior points
included), and the DINI domain the initial states are regridded from.

Usage (cartopy downloads coastlines on first use):

    uv run --with matplotlib python dev-utils/plot_domain.py \\
        <boundary datastore zarr, with latitude/longitude per grid_index> \\
        <DANRA grid zarr, with x/y/lat/lon> \\
        <DINI single_levels.zarr, with lat/lon> \\
        docs/clara_domain.png
"""
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

fp_boundary, fp_danra, fp_dini, fp_out = sys.argv[1:5]

# palette as for ANNA's boundary coverage plot (validated categorical slots 1
# and 2); text/outline ink from the reference text tokens
RING, INTERIOR = "#2a78d6", "#eb6834"
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# the DANRA projection (extra.projection in the interior datastore config)
DANRA_CRS = ccrs.LambertConformal(
    central_longitude=25.0,
    central_latitude=56.7,
    standard_parallels=(56.7, 56.7),
    globe=ccrs.Globe(semimajor_axis=6367470.0, semiminor_axis=6367470.0),
)

ds_b = xr.open_zarr(fp_boundary)
lat = ds_b.latitude.values
lon = ds_b.longitude.values
lon = np.where(lon > 180, lon - 360, lon)

danra = xr.open_zarr(fp_danra)[["lat", "lon"]].load()
xy = DANRA_CRS.transform_points(ccrs.PlateCarree(), lon, lat)
xs, ys = danra.x.values, danra.y.values
in_interior = (
    (xy[:, 0] >= xs.min())
    & (xy[:, 0] <= xs.max())
    & (xy[:, 1] >= ys.min())
    & (xy[:, 1] <= ys.max())
)

dini = xr.open_zarr(fp_dini)[["lat", "lon"]]
dini = dini.drop_vars([c for c in dini.coords if c not in "lat lon x y"])
dini = dini.load()


def perimeter(lat2d, lon2d):
    lo = np.concatenate(
        [lon2d[0, :], lon2d[:, -1], lon2d[-1, ::-1], lon2d[::-1, 0]]
    )
    la = np.concatenate(
        [lat2d[0, :], lat2d[:, -1], lat2d[-1, ::-1], lat2d[::-1, 0]]
    )
    return np.where(lo > 180, lo - 360, lo), la


proj = ccrs.LambertConformal(
    central_longitude=8, central_latitude=56, standard_parallels=(56, 56)
)
pc = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 8), facecolor=SURFACE)
ax = fig.add_axes([0.07, 0.07, 0.9, 0.82], projection=proj, facecolor=SURFACE)
ax.set_extent([-25, 38, 37, 72], crs=pc)
ax.add_feature(
    cfeature.LAND.with_scale("50m"), facecolor="#efeeea", edgecolor="none"
)
ax.add_feature(
    cfeature.COASTLINE.with_scale("50m"), edgecolor="#a3a29c", linewidth=0.5
)
gl = ax.gridlines(
    draw_labels=True,
    x_inline=False,
    y_inline=False,
    linewidth=0.4,
    color="#d6d5d0",
)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = dict(color=INK_2, fontsize=8)

ax.scatter(
    lon[~in_interior],
    lat[~in_interior],
    s=1.2,
    color=RING,
    transform=pc,
    linewidths=0,
    label=f"boundary point in the 400 km margin ({(~in_interior).sum():,})",
    rasterized=True,
)
ax.scatter(
    lon[in_interior],
    lat[in_interior],
    s=1.2,
    color=INTERIOR,
    transform=pc,
    linewidths=0,
    label=f"boundary point inside the interior ({in_interior.sum():,})",
    rasterized=True,
)

lo, la = perimeter(dini.lat.values, dini.lon.values)
ax.plot(lo, la, color=INK, linewidth=2, transform=pc)
lo, la = perimeter(danra.lat.values, danra.lon.values)
ax.plot(lo, la, color=INK_2, linewidth=2, linestyle=(0, (4, 3)), transform=pc)

# direct labels on the outlines
ax.text(
    -24,
    44.0,
    "DINI domain\n(initial states)",
    color=INK,
    fontsize=10,
    fontweight="bold",
    transform=pc,
)
ax.text(
    1.0,
    59.5,
    "DANRA interior\n(CLARA's grid)",
    color=INK_2,
    fontsize=9,
    fontweight="bold",
    transform=pc,
    ha="left",
    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2),
)

ax.legend(
    loc="lower right",
    markerscale=8,
    frameon=True,
    fontsize=9,
    facecolor=SURFACE,
    edgecolor="#d6d5d0",
    labelcolor=INK,
)
fig.text(
    0.07,
    0.91,
    "CLARA domain: DANRA interior and IFS boundary points\n"
    "(0.25°, 400 km margin, interior points included)",
    color=INK,
    fontsize=12,
    ha="left",
    va="bottom",
)
# NB: no bbox_inches="tight", which crops cartopy maps wrongly with the
# matplotlib version in the neural-lam environment
fig.savefig(fp_out, dpi=150, facecolor=SURFACE)
print(
    f"wrote {fp_out}: {lat.size} boundary points, {in_interior.sum()} "
    f"inside the interior"
)

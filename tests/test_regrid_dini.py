"""
Tests for the numerics and physics in `clara_inference.regrid_dini`.

    uv run pytest tests
"""

import numpy as np
import pytest
import xarray as xr

from clara_inference import regrid_dini as rd


def test_wind_rotation_round_trip():
    rng = np.random.default_rng(0)
    u, v, theta = rng.normal(size=(3, 50))
    u2, v2 = rd.earth_to_grid(*rd.grid_to_earth(u, v, theta), theta)
    np.testing.assert_allclose(u2, u, atol=1e-12)
    np.testing.assert_allclose(v2, v, atol=1e-12)


def test_wind_rotation_direction():
    # grid x-axis pointing north-east (theta = 45 deg): a wind along the
    # grid x-axis blows towards the north-east
    u_e, v_e = rd.grid_to_earth(1.0, 0.0, np.deg2rad(45))
    np.testing.assert_allclose([u_e, v_e], [np.sqrt(0.5), np.sqrt(0.5)])


def test_grid_x_axis_angle():
    # a regular lat/lon grid has its x-axis pointing east everywhere
    lon, lat = np.meshgrid(np.arange(-10, 10, 0.5), np.arange(40, 70, 0.5))
    np.testing.assert_allclose(rd.grid_x_axis_angle(lat, lon), 0.0, atol=1e-12)
    # a grid rotated by 30 deg (in a local tangent plane at the equator)
    angle = np.deg2rad(30)
    i, j = np.meshgrid(np.arange(20.0), np.arange(20.0))
    lon_r = 0.01 * (i * np.cos(angle) - j * np.sin(angle))
    lat_r = 0.01 * (i * np.sin(angle) + j * np.cos(angle))
    np.testing.assert_allclose(
        rd.grid_x_axis_angle(lat_r, lon_r), angle, atol=1e-4
    )


def test_bilinear_exact_for_linear_field():
    x, y = np.arange(0.0, 100.0, 2.0), np.arange(-50.0, 50.0, 2.0)
    xx, yy = np.meshgrid(x, y)
    field = 3.0 * xx - 2.0 * yy + 5.0
    rng = np.random.default_rng(1)
    x_tgt = rng.uniform(x[0], x[-1], size=(7, 9))
    y_tgt = rng.uniform(y[0], y[-1], size=(7, 9))
    interp = rd.BilinearInterpolator(x, y, x_tgt, y_tgt)
    np.testing.assert_allclose(interp(field), 3.0 * x_tgt - 2.0 * y_tgt + 5.0)
    assert interp(field).shape == (7, 9)


def test_bilinear_outside_domain():
    x, y = np.arange(10.0), np.arange(10.0)
    field = np.add.outer(y, x)
    x_tgt, y_tgt = np.array([5.0, 20.0]), np.array([5.0, 5.0])
    with pytest.raises(ValueError):
        rd.BilinearInterpolator(x, y, x_tgt, y_tgt, outside="error")
    interp = rd.BilinearInterpolator(x, y, x_tgt, y_tgt, outside="nearest")
    assert interp.n_outside == 1
    # outside point gets the value of the nearest edge point (x=9, y=5)
    np.testing.assert_allclose(interp(field), [10.0, 14.0])


def _fields(shape):
    """Random (..., y, x) fields as a dask-backed DataArray, 2D chunks."""
    dims = ("time", "pressure", "y", "x")[-len(shape) :]
    data = np.random.default_rng(2).normal(size=shape)
    da = xr.DataArray(data, dims=dims)
    return da.chunk({d: 1 for d in dims[:-2]})


def test_bilinear_leading_dims_and_lazy():
    x, y = np.arange(30.0), np.arange(20.0)
    rng = np.random.default_rng(3)
    x_tgt, y_tgt = rng.uniform(0, 29, (4, 5)), rng.uniform(0, 19, (4, 5))
    interp = rd.BilinearInterpolator(x, y, x_tgt, y_tgt)
    da = _fields((3, 2, 20, 30))
    expected = np.array([[interp(f2d) for f2d in f3d] for f3d in da.values])
    np.testing.assert_array_equal(interp(da.values), expected)
    lazy = interp.apply(da, ("y_out", "x_out"))
    assert lazy.chunks is not None and lazy.dims[-2:] == ("y_out", "x_out")
    np.testing.assert_array_equal(lazy.values, expected)



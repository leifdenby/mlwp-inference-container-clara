"""An ERA5 boundary laid out as an IFS forecast (hindcasts)."""
import datetime

import numpy as np
import pytest
import xarray as xr

from clara_inference import ifs_boundary as ib

H = datetime.timedelta(hours=1)
T0 = np.datetime64("1999-12-01T00:00", "ns")


def _create_era5(n_days: int = 5) -> xr.Dataset:
    """
    Create a WeatherBench2-like ERA5 dataset, 6-hourly from 1999-12-01, whose
    values are the hours since 1999-12-01 (to check the valid times).

    Parameters
    ----------
    n_days : int
        Number of days.

    Returns
    -------
    xr.Dataset
        The dataset, with the contract's variables and extra levels.
    """
    times = T0 + np.arange(4 * n_days) * np.timedelta64(6, "h")
    hours = ((times - T0) / np.timedelta64(1, "h")).astype("f4")
    lats = np.arange(-90.0, 90.1, 0.25)[::40]
    lons = np.arange(0.0, 360.0, 0.25)[::40]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    data_vars = {
        v: (
            ("time", "latitude", "longitude"),
            np.broadcast_to(
                hours[:, None, None], (times.size, lats.size, lons.size)
            ),
        )
        for v in ib.SL_VARS
    }
    for v in ib.PL_VARS:
        data_vars[v] = (
            ("time", "level", "latitude", "longitude"),
            np.broadcast_to(
                hours[:, None, None, None],
                (times.size, len(levels), lats.size, lons.size),
            ),
        )
    return xr.Dataset(
        data_vars,
        coords=dict(time=times, level=levels, latitude=lats, longitude=lons),
    )


def test_era5_as_forecast_matches_the_valid_times() -> None:
    analysis_time = datetime.datetime(1999, 12, 2, 21)
    ds_forecast = ib.era5_as_forecast(_create_era5(), analysis_time, 27 * H)
    # the latest 00/12 cycle at or before analysis time - 6 h (15Z)
    np.testing.assert_array_equal(
        ds_forecast.time.values, [np.datetime64("1999-12-02T12:00", "ns")]
    )
    ib.check_contract(ds_forecast)
    leads_h = ds_forecast.prediction_timedelta.values / np.timedelta64(1, "h")
    # last interior time 02T21 + 27 h + 2 * 3 h = 04T06, 42 h after the
    # cycle -> 7 steps, plus BOUNDARY_EXTRA_STEPS
    assert leads_h.tolist() == list(
        range(0, 6 * (7 + ib.BOUNDARY_EXTRA_STEPS) + 1, 6)
    )
    assert ds_forecast.level.values.tolist() == ib.LEVELS
    # each lead time is ERA5 at its valid time (36 h after T0 is 02T12)
    da_t2m = ds_forecast["2m_temperature"].isel(
        time=0, latitude=0, longitude=0
    )
    np.testing.assert_array_equal(da_t2m.values, 36 + leads_h)
    da_t = ds_forecast["temperature"].isel(time=0, latitude=0, longitude=0)
    np.testing.assert_array_equal(da_t.sel(level=850).values, 36 + leads_h)


@pytest.mark.parametrize(
    "analysis_time, cycle",
    [
        # 03/15 UTC: a 6 h boundary step before the first initial state
        (datetime.datetime(1999, 12, 3, 3), "1999-12-02T12:00"),
        (datetime.datetime(1999, 12, 3, 15), "1999-12-03T00:00"),
        (datetime.datetime(1999, 12, 3, 0), "1999-12-02T12:00"),
        (datetime.datetime(1999, 12, 3, 6), "1999-12-03T00:00"),
        (datetime.datetime(1999, 12, 3, 21), "1999-12-03T12:00"),
    ],
)
def test_era5_as_forecast_cycle(
    analysis_time: datetime.datetime, cycle: str
) -> None:
    ds_forecast = ib.era5_as_forecast(_create_era5(), analysis_time, 12 * H)
    assert ds_forecast.time.values[0] == np.datetime64(cycle, "ns")
    # neural-lam keeps the first initial state (no cropping)
    first_valid = ds_forecast.time.values[0]
    assert np.datetime64(analysis_time, "ns") >= first_valid + np.timedelta64(
        ib.LEAD_STEP
    )
    # and prepare_boundary_input picks the same cycle
    assert ib.select_cycle(ds_forecast.time.values, analysis_time) == (
        np.datetime64(cycle, "ns")
    )


def test_era5_as_forecast_missing_valid_times() -> None:
    with pytest.raises(ValueError, match="ERA5 has no valid times"):
        ib.era5_as_forecast(
            _create_era5(n_days=3), datetime.datetime(1999, 12, 2, 21), 27 * H
        )

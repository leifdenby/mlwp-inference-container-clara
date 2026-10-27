"""
Check that the operational boundary datastore configs (DINI, IFS; package
data in `anna_inference/boundary/`) give exactly the boundary features of the
ERA5 training boundary (`artifact/configs/era_7deg_model1_config.yaml`), and
follow the data contract documented in `ifs_7deg_model1_config.yaml`.

Synthetic zarr datasets following the contract are created for the IFS/DINI
boundaries (and in ERA5 layout for the training boundary), and each config is
run through `mllam_data_prep.create_dataset` (with `domain_cropping` disabled,
as that would require building the full DANRA interior dataset). The resulting
boundary features must match the ERA5 training boundary exactly, in number and
order, as that is what the checkpoint expects.

Requires a mllam-data-prep version supporting `domain_cropping` and
`lead_time` in derived variables, i.e. the one pinned in the ANNA
`pyproject.toml` (sadamov/mllam-data-prep@building-ml-lams):

    uv run pytest tests
"""
from pathlib import Path

import mllam_data_prep as mdp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from anna_inference.create_inference_dataset import (
    boundary_times,
    boundary_to_valid_time,
)
from anna_inference.model_configs import (
    BOUNDARY_SOURCES,
    boundary_config_path,
    inference_neural_lam_config,
    training_configs,
)

# the training configs in the repository (inputs to the inference artifact)
TRAINING_CONFIGS_DIR = Path(__file__).parent.parent / "artifact" / "configs"

FORECAST_ANALYSIS_TIME = "2026-09-26T12:00"
FORECAST_DURATION = pd.Timedelta("6h").to_pytimedelta()

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]
SURFACE_VARS = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
STATIC_VARS = ["land_sea_mask", "geopotential_at_surface"]
DERIVED_VARS = [
    "toa_radiation",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
N_FORCING_FEATURES = (
    len(SURFACE_VARS)
    + len(DERIVED_VARS)
    + len(PRESSURE_LEVEL_VARS) * len(LEVELS)
)

BOUNDARY_CONFIGS = {
    "era5": "era_7deg_model1_config.yaml",  # training
    "ifs": "ifs_7deg_model1_config.yaml",
    "dini": "dini_7deg_model1_config.yaml",
}


def _boundary_config(name):
    """Path of a boundary datastore config: training or operational."""
    if name == "era5":
        return TRAINING_CONFIGS_DIR / BOUNDARY_CONFIGS[name]
    return boundary_config_path(name)


# small regular lat/lon grid, a few points is enough
LATS = np.array([60.0, 59.75, 59.5])
LONS = np.array([10.0, 10.25, 10.5, 10.75])


def _spatial_coords():
    return dict(latitude=("latitude", LATS), longitude=("longitude", LONS))


def _make_forecast_contract_dataset(analysis_time, lead_hours):
    """
    Synthetic dataset following the IFS boundary contract (also used by the
    DINI boundary written by `anna_inference.regrid_dini`). An extra pressure level
    (500 hPa) is included to check that level selection works.
    """
    rng = np.random.default_rng(42)
    levels = sorted(LEVELS + [500])
    coords = dict(
        time=("time", [np.datetime64(analysis_time, "ns")]),
        prediction_timedelta=(
            "prediction_timedelta",
            pd.to_timedelta(lead_hours, unit="h").values,
        ),
        level=("level", levels, {"units": "hPa"}),
        **_spatial_coords(),
    )
    sfc_dims = ("time", "prediction_timedelta", "latitude", "longitude")
    pl_dims = (
        "time",
        "prediction_timedelta",
        "level",
        "latitude",
        "longitude",
    )
    sfc_shape = (1, len(lead_hours), len(LATS), len(LONS))
    pl_shape = (1, len(lead_hours), len(levels), len(LATS), len(LONS))

    data_vars = {}
    for var in SURFACE_VARS:
        data_vars[var] = (sfc_dims, rng.normal(size=sfc_shape))
    for var in PRESSURE_LEVEL_VARS:
        data_vars[var] = (pl_dims, rng.normal(size=pl_shape))
    # statics must not carry time/lead-time/level dims
    for var in STATIC_VARS:
        data_vars[var] = (
            ("latitude", "longitude"),
            rng.normal(size=(len(LATS), len(LONS))),
        )

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _make_era5_layout_dataset(times):
    """Synthetic dataset in the (WeatherBench2) ERA5 layout used for training."""
    rng = np.random.default_rng(42)
    coords = dict(
        time=("time", pd.to_datetime(times).values),
        level=("level", LEVELS),
        **_spatial_coords(),
    )
    data_vars = {}
    for var in SURFACE_VARS:
        data_vars[var] = (
            ("time", "latitude", "longitude"),
            rng.normal(size=(len(times), len(LATS), len(LONS))),
        )
    for var in PRESSURE_LEVEL_VARS:
        data_vars[var] = (
            ("time", "level", "latitude", "longitude"),
            rng.normal(size=(len(times), len(LEVELS), len(LATS), len(LONS))),
        )
    for var in STATIC_VARS:
        data_vars[var] = (
            ("latitude", "longitude"),
            rng.normal(size=(len(LATS), len(LONS))),
        )
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _create_boundary_dataset(config_name, fp_input):
    config = mdp.Config.from_yaml_file(_boundary_config(config_name))
    for input_config in config.inputs.values():
        input_config.path = str(fp_input)
    # cropping needs the full interior (DANRA) dataset, which is out of scope
    # for this test
    config.output.domain_cropping = None
    return mdp.create_dataset(config=config)


@pytest.fixture(scope="module")
def boundary_datasets(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("anna_boundary")

    fp_era5 = tmp_path / "era5_layout.zarr"
    times = pd.date_range("2000-01-01T00:00", periods=4, freq="6h")
    ds_era5 = _make_era5_layout_dataset(times=times)
    ds_era5.to_zarr(fp_era5)

    datasets = {}
    for name in ["ifs", "dini"]:
        # the IFS/DINI configs are used as-is, only the input paths change.
        # They read the forecast after conversion to valid-time layout
        ds = _make_forecast_contract_dataset(
            analysis_time=FORECAST_ANALYSIS_TIME, lead_hours=[0, 6, 12]
        )
        ds_valid = boundary_to_valid_time(
            ds,
            pd.Timestamp(FORECAST_ANALYSIS_TIME).to_pydatetime(),
            FORECAST_DURATION,
        )
        fp = tmp_path / f"{name}.zarr"
        ds_valid.to_zarr(fp)
        datasets[name] = _create_boundary_dataset(name, fp)

    # training config covers 2000-2020, the synthetic data only a day
    config = mdp.Config.from_yaml_file(_boundary_config("era5"))
    for input_config in config.inputs.values():
        input_config.path = str(fp_era5)
    config.output.domain_cropping = None
    config.output.coord_ranges["time"].start = str(times[0])
    config.output.coord_ranges["time"].end = str(times[-1])
    for split in config.output.splitting.splits.values():
        split.start = str(times[0])
        split.end = str(times[-1])
    datasets["era5"] = mdp.create_dataset(config=config)

    return datasets


@pytest.mark.parametrize("name", list(BOUNDARY_CONFIGS))
def test_boundary_config_loads(name):
    config = mdp.Config.from_yaml_file(_boundary_config(name))
    cropping = config.output.domain_cropping
    assert cropping.margin_width_degrees == 7.19
    assert cropping.include_interior_points is False
    # cropped around the training interior (create_inference_dataset points
    # the operational configs at the inference interior config)
    assert cropping.interior_dataset_config_path == "danra_model1_config.yaml"


def test_operational_boundary_configs_are_packaged():
    assert set(BOUNDARY_SOURCES) == {"dini", "ifs"}
    for name in BOUNDARY_SOURCES:
        assert boundary_config_path(name).name == BOUNDARY_CONFIGS[name]
        assert boundary_config_path(name).exists()
    with pytest.raises(KeyError):
        boundary_config_path("era5")  # the training boundary isn't one


def test_interior_config_loads():
    mdp.Config.from_yaml_file(
        TRAINING_CONFIGS_DIR / "danra_model1_config.yaml"
    )


def test_training_configs_resolve():
    # model.yaml -> training neural-lam config -> datastore configs
    configs = training_configs(TRAINING_CONFIGS_DIR)
    assert configs.neural_lam_config == "7deg_config_era5.yaml"
    assert configs.interior_datastore == "danra_model1_config.yaml"
    assert configs.boundary_datastore == BOUNDARY_CONFIGS["era5"]
    for fn in [
        configs.neural_lam_config,
        configs.interior_datastore,
        configs.boundary_datastore,
    ]:
        assert (TRAINING_CONFIGS_DIR / fn).exists()
    # nothing but the training configs and the model description
    assert sorted(p.name for p in TRAINING_CONFIGS_DIR.iterdir()) == sorted(
        [
            "model.yaml",
            configs.neural_lam_config,
            configs.interior_datastore,
            configs.boundary_datastore,
        ]
    )


@pytest.mark.parametrize("name", list(BOUNDARY_SOURCES))
def test_inference_neural_lam_config(name):
    configs = training_configs(TRAINING_CONFIGS_DIR)
    training = yaml.safe_load(
        (TRAINING_CONFIGS_DIR / configs.neural_lam_config).read_text()
    )
    nl_config = inference_neural_lam_config(
        training,
        interior_config="danra_model1_config.yaml",
        boundary_config=BOUNDARY_CONFIGS[name],
        stats_config=configs.boundary_datastore,
    )
    # as the formerly stored 7deg_config_{dini,ifs}.yaml: the training config
    # with the operational boundary, normalised with the ERA5 statistics
    expected = yaml.safe_load(yaml.safe_dump(training))
    expected["datastore_boundary"]["config_path"] = BOUNDARY_CONFIGS[name]
    expected["datastore_boundary"]["overload_stats_path"] = BOUNDARY_CONFIGS[
        "era5"
    ]
    assert nl_config == expected
    # the training config itself is left alone
    assert "overload_stats_path" not in training["datastore_boundary"]


@pytest.mark.parametrize("name", list(BOUNDARY_CONFIGS))
def test_boundary_feature_counts(boundary_datasets, name):
    ds = boundary_datasets[name]
    assert ds.forcing_feature.size == N_FORCING_FEATURES == 58
    assert ds.static_feature.size == len(STATIC_VARS) == 2
    assert ds.grid_index.size == len(LATS) * len(LONS)


@pytest.mark.parametrize("name", ["ifs", "dini"])
def test_boundary_features_match_training(boundary_datasets, name):
    ds_train = boundary_datasets["era5"]
    ds = boundary_datasets[name]
    assert list(ds.forcing_feature.values) == list(
        ds_train.forcing_feature.values
    )
    assert list(ds.static_feature.values) == list(
        ds_train.static_feature.values
    )


@pytest.mark.parametrize("name", ["ifs", "dini"])
def test_forecast_boundary_dims(boundary_datasets, name):
    ds = boundary_datasets[name]
    # same layout as the ERA5 training boundary: valid times
    assert set(ds.forcing.dims) == set(boundary_datasets["era5"].forcing.dims)
    assert "analysis_time" not in ds.dims
    # 12Z + 6h forecast: boundary 06Z (padded) .. 06Z next day (padded)
    assert ds.time.size == 5
    assert not bool(ds.forcing.isnull().any())


def _dt(s):
    return pd.Timestamp(s).to_pydatetime()


def _hours(h):
    return pd.Timedelta(hours=h).to_pytimedelta()


@pytest.mark.parametrize(
    "analysis_time, duration_h, used, required",
    [
        # synoptic analysis time: the boundary times used start at it
        (
            "2026-10-01T00:00",
            6,
            ("00", "12"),
            ("2026-09-30T18", "2026-10-01T18"),
        ),
        ("2026-10-01T00:00", 18, ("00", "2026-10-02T00"), None),
        # 03Z: the first prediction (09Z) uses 00/06/12Z
        ("2026-10-01T03:00", 6, ("00", "12"), None),
        # forecast ending off the boundary grid (+9h -> 12Z): 06/12/18Z
        ("2026-10-01T03:00", 9, ("00", "18"), None),
    ],
)
def test_boundary_times(analysis_time, duration_h, used, required):
    def full(t):
        return _dt(t if len(t) > 2 else f"2026-10-01T{t}:00")

    u, r = boundary_times(_dt(analysis_time), _hours(duration_h))
    assert (u[0], u[-1]) == (full(used[0]), full(used[1]))
    # 6-hourly at 00/06/12/18 UTC, used within required
    assert all(t.hour % 6 == 0 and t.minute == 0 for t in r)
    assert set(u) <= set(r)
    # neural-lam's coverage check: one boundary step before the analysis time
    # and after the last interior time (forecast + 2 interior steps)
    t0 = _dt(analysis_time)
    assert r[0] <= t0 - _hours(6)
    assert r[-1] >= t0 + _hours(duration_h) + _hours(6) + _hours(6)
    if required is not None:
        assert (r[0], r[-1]) == (_dt(required[0]), _dt(required[1]))


def test_boundary_to_valid_time_pads_with_copies():
    ds = _make_forecast_contract_dataset(
        analysis_time="2026-10-01T00:00", lead_hours=[0, 6, 12, 18]
    )
    ds_valid = boundary_to_valid_time(ds, _dt("2026-10-01T00:00"), _hours(6))
    times = [str(t)[:13] for t in ds_valid.time.values]
    assert times == [
        "2026-09-30T18",
        "2026-10-01T00",
        "2026-10-01T06",
        "2026-10-01T12",
        "2026-10-01T18",
    ]
    t2m = ds_valid["2m_temperature"]
    # padded ends are copies of the nearest used time; the used times are
    # the forecast's own data, even where it has more lead times
    np.testing.assert_array_equal(t2m.isel(time=0), t2m.isel(time=1))
    np.testing.assert_array_equal(
        t2m.isel(time=4),
        ds["2m_temperature"].isel(time=0, prediction_timedelta=2),
    )
    np.testing.assert_array_equal(
        t2m.isel(time=2),
        ds["2m_temperature"].isel(time=0, prediction_timedelta=1),
    )
    assert ds_valid["land_sea_mask"].dims == ("latitude", "longitude")
    assert ds_valid.attrs["boundary_times_padded"] == (
        "2026-09-30T18:00, 2026-10-01T18:00"
    )


def test_boundary_to_valid_time_older_cycle():
    # e.g. IFS: the 00Z cycle for a 06Z ANNA run
    ds = _make_forecast_contract_dataset(
        analysis_time="2026-10-01T00:00", lead_hours=list(range(0, 31, 6))
    )
    ds_valid = boundary_to_valid_time(ds, _dt("2026-10-01T06:00"), _hours(18))
    assert ds_valid.attrs["boundary_cycle"] == "2026-10-01T00:00:00"
    assert ds_valid.attrs["boundary_times_used"] == (
        "2026-10-01T06:00 .. 2026-10-02T06:00"
    )


def test_boundary_to_valid_time_from_zarr(tmp_path):
    # as in operation: the boundary opened from a zarr store with its lead
    # times in chunks of 3 (as zarr picks for regrid_dini's boundary box).
    # The padded times must not leave uneven chunks, which zarr can't store
    ds = _make_forecast_contract_dataset(
        analysis_time="2026-10-01T06:00", lead_hours=[0, 6, 12, 18, 24]
    )
    fp = tmp_path / "boundary.zarr"
    ds.chunk(prediction_timedelta=3).to_zarr(fp)
    ds_valid = boundary_to_valid_time(
        xr.open_zarr(fp), _dt("2026-10-01T06:00"), _hours(18)
    )
    fp_valid = tmp_path / "boundary_valid.zarr"
    ds_valid.to_zarr(fp_valid)
    xr.testing.assert_equal(xr.open_zarr(fp_valid), ds_valid)
    assert ds_valid.attrs["boundary_times_padded"] == (
        "2026-10-01T00:00, 2026-10-02T12:00"
    )


def test_boundary_to_valid_time_missing():
    # DINI boundary from the 03Z run: no data at 00Z, which a 03Z forecast
    # needs
    ds = _make_forecast_contract_dataset(
        analysis_time="2026-10-01T03:00", lead_hours=[0, 6, 12]
    )
    with pytest.raises(ValueError, match="has no data at"):
        boundary_to_valid_time(ds, _dt("2026-10-01T03:00"), _hours(6))
    # a cycle after the analysis time can't be used
    with pytest.raises(ValueError, match="no boundary cycle"):
        boundary_to_valid_time(ds, _dt("2026-10-01T00:00"), _hours(6))

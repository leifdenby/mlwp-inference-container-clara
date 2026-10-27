"""
Interim CLARA statistics datastores from the ANNA inference artifact.

ANNA's statistics (`danra_model1_config`, `era_7deg_model1_config`) have the
same features in the same order as CLARA's statistics datastores
(`danra_model1_config`, `overlapping_era_400km_model1_config`), and the ERA5
ones are computed over the same ERA5 subset box (mllam-data-prep computes the
statistics before the domain cropping). They are over ANNA's training period
(2000-01-01..2018-10-29) though, while CLARA's were over 2016-01-01..2019-10-22
(the split recorded in the checkpoint's pickled stats datastore), so they
approximate CLARA's statistics.

The `splits` variable (which neural-lam reads) is taken from a CLARA
statistics datastore made with `compute_stats` (e.g. the placeholder ones).

Usage:
    python dev-utils/stats_from_anna.py <anna-artifact>/stats <clara-stats-dir> <output-dir>
"""
import sys
from pathlib import Path

import xarray as xr

PAIRS = {
    "danra_model1_config": "danra_model1_config",
    "era_7deg_model1_config": "overlapping_era_400km_model1_config",
}


def main():
    anna_dir, clara_dir, output_dir = (Path(p) for p in sys.argv[1:4])
    output_dir.mkdir(parents=True, exist_ok=False)
    for anna_name, clara_name in PAIRS.items():
        ds_anna = xr.open_zarr(anna_dir / f"{anna_name}.stats.zarr").load()
        ds_clara = xr.open_zarr(clara_dir / f"{clara_name}.zarr").load()
        for coord in [c for c in ds_anna.coords if c.endswith("_feature")]:
            if list(ds_anna[coord].values) != list(ds_clara[coord].values):
                raise ValueError(f"{anna_name}/{clara_name}: {coord} differs")
        ds = ds_anna.copy()
        ds["splits"] = ds_clara["splits"]
        ds.attrs = dict(
            ds_anna.attrs,
            interim_source=f"ANNA artifact {anna_dir / anna_name}.stats.zarr",
            interim_note=(
                "ANNA's training period (2000-01-01..2018-10-29), not "
                "CLARA's (2016-01-01..2019-10-22): an approximation"
            ),
        )
        ds.to_zarr(output_dir / f"{clara_name}.zarr", mode="w", consolidated=True)
        print(f"wrote {output_dir / clara_name}.zarr")


if __name__ == "__main__":
    main()

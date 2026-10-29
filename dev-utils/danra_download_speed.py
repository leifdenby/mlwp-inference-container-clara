"""
Download speed from DANRA's object store, with N concurrent downloads of
whole chunk files (as `danra_stats`' file cache downloads them), to tell
whether `danra_stats` is limited by the download.

Usage:
    uv run python dev-utils/danra_download_speed.py <tmp dir> [N ...]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec

BASE = "https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/pressure_levels.zarr/"
# time chunk 90, all 14 levels and 6 spatial chunks of t: 84 files, ~3 GB
FILES = [f"t/90.{p}.{y}.{x}" for p in range(14) for y in range(2) for x in range(3)]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("https")
    files = iter(FILES)
    for n in [int(a) for a in sys.argv[2:]] or [1, 4, 16]:
        # 2 files per download slot, new ones each time (no caching)
        names = [next(files) for _ in range(2 * n)]

        def get(name):
            fp = out / name.replace("/", "_")
            fs.get(BASE + name, str(fp))
            size = fp.stat().st_size
            fp.unlink()
            return size

        t0 = time.time()
        with ThreadPoolExecutor(n) as pool:
            total = sum(pool.map(get, names))
        dt = time.time() - t0
        print(
            f"{n:2d} concurrent: {total / 1e6:5.0f} MB in {dt:5.1f} s = "
            f"{total / 1e6 / dt:6.1f} MB/s"
        )


if __name__ == "__main__":
    main()

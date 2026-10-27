"""
Write the git pins of pyproject.toml as pip requirement files, for the
container build (`uv pip install --no-sources`, which uses the base image's
torch rather than the cu130 index of [tool.uv.sources]):

- <out>/requirements.txt: the [tool.uv.sources] git sources as direct
  references (e.g. neural-lam from leifdenby/real-prob-lam)
- <out>/overrides.txt: [tool.uv] override-dependencies (mllam-data-prep)

Usage: python scripts/container_pins.py pyproject.toml <out-dir>
"""
import sys
import tomllib
from pathlib import Path


def git_requirement(name, source):
    url = f"git+{source['git']}@{source['rev']}"
    if "subdirectory" in source:
        url += f"#subdirectory={source['subdirectory']}"
    return f"{name} @ {url}"


def pins(pyproject):
    uv = pyproject["tool"]["uv"]
    requirements = [
        git_requirement(name, source)
        for name, source in uv.get("sources", {}).items()
        if isinstance(source, dict) and "git" in source
    ]
    return requirements, list(uv.get("override-dependencies", []))


def main():
    fp_pyproject, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    requirements, overrides = pins(tomllib.loads(fp_pyproject.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "requirements.txt").write_text("\n".join(requirements) + "\n")
    (out_dir / "overrides.txt").write_text("\n".join(overrides) + "\n")
    for line in requirements + overrides:
        print(line)


if __name__ == "__main__":
    main()

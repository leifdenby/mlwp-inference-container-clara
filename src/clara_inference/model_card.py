"""
CLARA's model card: `MODEL_CARD.md` generated from `artifact/model-card.yaml`
(the model, and its input and output variables, grouped by height, single
and pressure levels, with CF standard names).

The YAML is the single source. The state variables' training loss weights
are taken from the training neural-lam config (`artifact/configs/`), and
`tests/test_model_card.py` checks the variables against the training
datastore configs.

Usage:
    uv run python -m clara_inference.model_card           # write MODEL_CARD.md
    uv run python -m clara_inference.model_card --check   # is it up to date?
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

from clara_inference import model_configs as mc

REPO = Path(__file__).parents[2]
CARD = REPO / "artifact" / "model-card.yaml"
CONFIGS = REPO / "artifact" / "configs"
OUTPUT = REPO / "MODEL_CARD.md"

LEVEL_TYPES = ["height", "single", "pressure"]
CF_VALUES = ["yes", "partial", "no"]
REQUIRED = ["name", "long_name", "units", "level_type", "standard_name", "cf"]
# the datastore (interior/boundary) and category of each group
GROUPS = {
    "state": ("interior", "state"),
    "interior_forcing": ("interior", "forcing"),
    "interior_static": ("interior", "static"),
    "boundary_forcing": ("boundary", "forcing"),
    "boundary_static": ("boundary", "static"),
}


def load(fp=CARD):
    """The model card, checked (ValueError on problems)."""
    card = yaml.safe_load(Path(fp).read_text())
    problems = []
    if list(card["groups"]) != list(GROUPS):
        problems.append(f"groups must be {list(GROUPS)}")
    for group in card["groups"].values():
        for v in group["variables"]:
            name = v.get("name", "?")
            problems += [
                f"{name}: no {key}" for key in REQUIRED if key not in v
            ]
            if v.get("level_type") not in LEVEL_TYPES:
                problems.append(f"{name}: level_type not in {LEVEL_TYPES}")
            if (v.get("level_type") == "single") == bool(v.get("levels")):
                problems.append(
                    f"{name}: levels are needed for (only) height and "
                    "pressure levels"
                )
            if v.get("cf") not in CF_VALUES:
                problems.append(f"{name}: cf not in {CF_VALUES}")
            if (v.get("cf") == "no") != (v.get("standard_name") is None):
                problems.append(
                    f"{name}: standard_name must be null exactly when cf "
                    "is no"
                )
            if v.get("cf") in ("partial", "no") and not v.get("cf_note"):
                problems.append(f"{name}: cf {v.get('cf')} needs a cf_note")
    if problems:
        raise ValueError(f"model card {fp}: " + "; ".join(problems))
    return card


def features(variables):
    """
    The datastore feature names of `variables`, in order: pressure-level
    variables per level (`{name}{level}`, as the datastore configs'
    name_format), the others by name.
    """
    return [
        f"{v['name']}{level}"
        for v in variables
        for level in (v["levels"] if v["level_type"] == "pressure" else [""])
    ]


def loss_weights(configs_dir=CONFIGS):
    """The state feature loss weights of the training neural-lam config."""
    training = mc.training_configs(configs_dir)
    config = yaml.safe_load(
        (Path(configs_dir) / training.neural_lam_config).read_text()
    )
    return config["training"]["state_feature_weighting"]["weights"]


def _text(s):
    """One line of table text."""
    return re.sub(r"\s+", " ", str(s)).strip().replace("|", "\\|")


def _levels(v):
    if v["level_type"] == "height":
        return ", ".join(f"{h} m" for h in v["levels"])
    if v["level_type"] == "pressure":
        return ", ".join(str(p) for p in v["levels"]) + " hPa"
    return ""


def _cf(v):
    note = f": {_text(v['cf_note'])}" if v.get("cf_note") else ""
    return {"yes": "yes", "partial": "**partial**", "no": "**no**"}[
        v["cf"]
    ] + note


def _weight(v, weights):
    values = {weights[f] for f in features([v])}
    return ", ".join(f"{w:g}" for w in sorted(values))


def _table(variables, weights=None):
    columns = [
        "Variable",
        "Description",
        "Units",
        "Levels",
        "CF standard name",
        "CF",
        "Source",
        "Notes",
    ]
    if weights is not None:
        columns.insert(-1, "Loss weight")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for v in variables:
        row = [
            f"`{v['name']}`",
            _text(v["long_name"]),
            f"`{v['units']}`",
            _levels(v),
            f"`{v['standard_name']}`" if v["standard_name"] else "(none)",
            _cf(v),
            _text(v.get("source", "")),
            _text(v.get("notes", "")),
        ]
        if weights is not None:
            row.insert(-1, _weight(v, weights))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render(card, weights):
    """MODEL_CARD.md of `card`, with the state's training loss weights."""
    m = card["model"]
    paper = m["paper"]
    out = [
        f"# {m['name']} model card",
        "",
        "<!-- generated from artifact/model-card.yaml by "
        "`python -m clara_inference.model_card`: edit that file instead -->",
        "",
        f"**{m['name']}** ({m['full_name']}). {_text(m['summary'])}",
        "",
        "## Model",
        "",
        f"- **Paper:** {_text(paper['title'])}, {paper['authors']}, "
        f"[arXiv:{paper['arxiv']}](https://arxiv.org/abs/{paper['arxiv']})",
        f"- **Architecture:** {_text(m['architecture'])}",
        "- **Training data:**",
    ]
    out += [
        f"  - {key}: {_text(value)}"
        for key, value in m["training_data"].items()
    ]
    out += [
        f"- **Intended use:** {_text(m['intended_use'])}",
        "",
        "## Time, ensemble and grids",
        "",
    ]
    out += [
        f"- **{key.replace('_', ' ').capitalize()}:** {_text(value)}"
        for key, value in card["time"].items()
    ]
    out += [f"- **Ensemble:** {_text(card['ensemble'])}"]
    out += [
        f"- **{key.capitalize()} grid:** {_text(value)}"
        for key, value in card["grids"].items()
    ]
    out += ["", "## Output files", ""]
    out += [
        f"- `{key}`: {_text(value)}" if key.endswith(".zarr")
        else f"- {_text(value).capitalize()}."
        for key, value in card["output_files"].items()
    ]
    out += [
        "",
        "## Reading the tables",
        "",
        f"- **CF standard name:** from the CF standard name table, version "
        f"{card['cf_standard_name_table']}. (none) means that no standard "
        "name exists for the variable.",
        "- **CF:** whether the variable follows CF as stored (in the output "
        "files for the state, in the datastores for the other groups). "
        "**yes**: the "
        "standard name and units describe it. **partial**: a standard name "
        "applies, but something about the variable as stored differs. "
        "**no**: there is no standard name. The note says why, or where the "
        "source datasets' own metadata differ.",
        "- **Source:** where the operational inputs come from, and how they "
        "are converted.",
        "- **Loss weight:** the state feature's weight in the training loss "
        "(per level for pressure levels).",
        "- Tables are grouped by level type:",
    ]
    out += [
        f"  - {lt['title']}: {_text(lt['description'])}"
        for lt in card["level_types"].values()
    ]
    for key, group in card["groups"].items():
        out += ["", f"## {group['title']}", "", _text(group["description"])]
        for level_type in LEVEL_TYPES:
            variables = [
                v for v in group["variables"]
                if v["level_type"] == level_type
            ]
            if not variables:
                continue
            title = card["level_types"][level_type]["title"]
            out += ["", f"### {title}", ""]
            if key == "state":
                files = sorted({v["output_file"] for v in variables})
                out += [
                    "Output file: "
                    + ", ".join(f"`{fn}`" for fn in files)
                    + ".",
                    "",
                ]
            out += _table(variables, weights if key == "state" else None)
    out += ["", "## CF summary", ""]
    for key, group in card["groups"].items():
        counts = {
            cf: [v["name"] for v in group["variables"] if v["cf"] == cf]
            for cf in CF_VALUES
        }
        text = f"- **{group['title']}:** {len(counts['yes'])} yes"
        for cf in ["partial", "no"]:
            if counts[cf]:
                names = ", ".join(f"`{n}`" for n in counts[cf])
                text += f"; {len(counts[cf])} {cf} ({names})"
        out.append(text)
    out += ["", "## Not available", ""]
    out += [
        f"- **{item['name']}:** {_text(item['note'])}"
        for item in card["not_available"]
    ]
    out += ["", "## Limitations", ""]
    out += [f"- {_text(item)}" for item in m["limitations"]]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--card", type=Path, default=CARD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only check that the output is up to date (exit code 1 if not)",
    )
    args = parser.parse_args()
    text = render(load(args.card), loss_weights())
    if args.check:
        if not args.output.exists() or args.output.read_text() != text:
            print(f"{args.output} is out of date: regenerate it")
            sys.exit(1)
        print(f"{args.output} is up to date")
        return
    args.output.write_text(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

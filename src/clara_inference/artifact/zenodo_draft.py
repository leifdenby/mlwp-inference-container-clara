"""
Create (or update) a DRAFT Zenodo deposition for CLARA: the model checkpoint
(`--kind checkpoint`) or the inference package (`--kind package`).

As for ANNA, the checkpoint and the inference package are separate records:
the package (configs, statistics, grids, graph; `assemble --zip`) refers to
the checkpoint's record by DOI, and the container downloads both by URL and
md5. Draft the checkpoint first: its DOI is reserved with the draft, and
goes into `artifact/configs/model.yaml` (with the record id, the url and the
md5 printed here) before assembling the package.

Nothing is ever published by this script: review the drafts in the Zenodo
web UI (and share them for review there) and publish them there (publishing
is irreversible). The package refuses an artifact with placeholder (e.g.
interim) statistics.

The Zenodo personal access token (scope `deposit:write`) is read from
`--token-file`, or else from the `ZENODO_TOKEN` environment variable, and is
not stored anywhere.

Usage:
    # review the metadata without contacting Zenodo
    python -m clara_inference.artifact.zenodo_draft --kind checkpoint \\
        --file artifact/build/clara-local/clara_model.ckpt --dry-run
    # rehearse on sandbox.zenodo.org (needs a sandbox token)
    python -m clara_inference.artifact.zenodo_draft --kind checkpoint \\
        --file .../clara_model.ckpt --token-file ~/.zenodo-sandbox-token \\
        --sandbox
    # create the draft on zenodo.org
    python -m clara_inference.artifact.zenodo_draft --kind checkpoint \\
        --file .../clara_model.ckpt --token-file ~/.zenodo-token
    # the package, once model.yaml has the checkpoint's DOI and the
    # artifact has the full training statistics
    python -m clara_inference.artifact.zenodo_draft --kind package \\
        --file artifact/build/clara-<date>.zip --token-file ~/.zenodo-token
    # update an existing draft (replace the file, update the metadata)
    ... --deposition-id 1234567
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import yaml

from clara_inference.fetch_checkpoint import _md5

ZENODO = "https://zenodo.org"
ZENODO_SANDBOX = "https://sandbox.zenodo.org"
REPO_URL = "https://github.com/leifdenby/mlwp-inference-container-clara"

PAPER_TITLE = (
    "CRPS-LAM: Probabilistic Regional Weather Forecasting with Continuous "
    "Ranked Probability Score"
)
PAPER_ARXIV = "arXiv:2510.09484"
# as in the paper (arXiv v3)
PAPER_AUTHORS = [
    {"name": "Wikingsson, Erik", "affiliation": "Linköping University"},
    {"name": "Oskarsson, Joel", "affiliation": "ETH AI Center"},
    {
        "name": "Landelius, Tomas",
        "affiliation": "SMHI; Linköping University",
    },
    {"name": "Lindsten, Fredrik", "affiliation": "Linköping University"},
]
PACKAGE_AUTHORS = [
    {"name": "Denby, Leif", "affiliation": "Danish Meteorological Institute"},
]
REVIEW_NOTES = [
    "creators: names and affiliations as in the paper (arXiv v3), no ORCIDs",
    "license: cc-by-4.0 is a placeholder, to be decided by the authors",
]
KEYWORDS = [
    "machine learning weather prediction",
    "limited area model",
    "probabilistic forecasting",
    "CRPS",
    "neural-lam",
    "DANRA",
    "IFS",
]


def _paper_relation():
    return {
        "identifier": PAPER_ARXIV,
        "relation": "isSupplementTo",
        "resource_type": "publication-preprint",
        "scheme": "arxiv",
    }


def checkpoint_metadata(fp_ckpt, model, version):
    """The checkpoint record's metadata (`model`: model.yaml)."""
    training = model["training"]
    args = model["train_model_args"]
    description = f"""
<p>Model weights of <b>CLARA</b> (CRPS-LAM Rapid Atmospheric ensembles): the
probabilistic limited area model CRPS-LAM of
<a href="https://arxiv.org/abs/2510.09484">Wikingsson et al.,
"{PAPER_TITLE}"</a>, trained on the DANRA reanalysis (interior) and
fine-tuned with an IFS forecast boundary.</p>

<p>The model is neural-lam's <code>crps</code> model with the hybrid graph +
UNet backbone <code>graph_unet2</code> and noise injection, trained with the
CRPS. It predicts the 55 state variables of DANRA (on its 589 x 789, 2.5 km
grid) in 3-hourly steps, with a 400 km IFS boundary (6-hourly lead times of
the 00/12 UTC cycles, normalised with ERA5 statistics). Training statistics
period: 2016-01-01 to 2019-10-22.</p>

<p>File: <code>{fp_ckpt.name}</code>, a PyTorch Lightning checkpoint
(training run <code>{training['run_name']}</code>, epoch {training['epoch']},
step {training['global_step']}) reduced to the weights
(<code>state_dict</code>) and the training arguments and config as plain
data, so it loads with <code>torch.load(..., weights_only=True)</code>.
Optimizer state and the pickled training datastores were removed; the
weights are unchanged.</p>

<p>To run the model you also need the inference package (training configs,
normalisation statistics, grids and graph), published separately, and
neural-lam as pinned in
<a href="{REPO_URL}">mlwp-inference-container-clara</a>, which runs CLARA
operationally from DINI initial states and an IFS boundary. Model arguments:
<code>{' '.join(str(a) for a in args)}</code>.</p>
"""
    return {
        "upload_type": "other",
        "title": (
            "CLARA: CRPS-LAM model weights (DANRA interior, IFS boundary "
            "fine-tune)"
        ),
        "creators": PAPER_AUTHORS,
        "description": " ".join(description.split()),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": version,
        "keywords": KEYWORDS,
        "related_identifiers": [_paper_relation()],
        "prereserve_doi": True,
    }


def package_metadata(fp_zip, version):
    """The inference package record's metadata, partly from the zip."""
    with zipfile.ZipFile(fp_zip) as zf:
        artifact = yaml.safe_load(zf.read("artifact.yaml"))
        model = yaml.safe_load(zf.read("configs/model.yaml"))
    if artifact.get("placeholder_stats"):
        raise ValueError(
            f"{fp_zip} has placeholder statistics (artifact.yaml): assemble "
            "it with the full training statistics first"
        )
    ckpt = model["checkpoint"]
    if not ckpt.get("doi"):
        raise ValueError(
            "the package's configs/model.yaml has no checkpoint DOI: draft "
            "the checkpoint (--kind checkpoint) and fill in model.yaml first"
        )
    stats = "".join(
        f"<li><code>{name}</code>: {attrs.get('description', '')} "
        f"({attrs.get('split_start', attrs.get('statistics_period', ''))}"
        f"{' to ' + attrs['split_end'] if 'split_end' in attrs else ''})"
        "</li>"
        for name, attrs in artifact["stats"].items()
    )
    description = f"""
<p>Inference package of <b>CLARA</b> (CRPS-LAM Rapid Atmospheric ensembles),
the CRPS-LAM model of <a href="https://arxiv.org/abs/2510.09484">Wikingsson
et al., "{PAPER_TITLE}"</a> trained on DANRA and fine-tuned with an IFS
forecast boundary: everything needed to run the model as it was trained,
except the weights.</p>

<p><b>This package does not contain the model weights.</b> They are
published separately: <a href="https://doi.org/{ckpt['doi']}">{ckpt['doi']}
</a> (<code>{ckpt['artifact_path']}</code>, md5 <code>{ckpt['md5']}</code>).
</p>

<p>Contents:</p>
<ul>
<li><code>configs/</code>: the neural-lam config and the mllam-data-prep
datastore configs the model was trained with (DANRA interior, IFS
boundary, and the statistics datastores), and <code>model.yaml</code>
describing the model (checkpoint, graph, neural-lam arguments).</li>
<li><code>configs/*.zarr</code>: the training (normalisation) statistics,
recomputed exactly from the public data (DANRA v0.5.0, WeatherBench2 ERA5)
over the training period, as mllam-data-prep computed them:<ul>{stats}</ul>
</li>
<li><code>graphs/</code>: the graph the model was trained with.</li>
<li><code>grids/</code>: the DANRA grid and static fields, and the
{artifact['boundary_points']} boundary points (0.25 deg) with the ERA5
static fields.</li>
<li><code>README.md</code>, <code>artifact.yaml</code>: description and
provenance.</li>
</ul>

<p>Running the model from operational data is done by
<a href="{REPO_URL}">mlwp-inference-container-clara</a>, which runs it from
DINI initial states and an IFS boundary.</p>
"""
    return {
        "upload_type": "dataset",
        "title": "CLARA inference package for the CRPS-LAM model",
        "creators": PAPER_AUTHORS + PACKAGE_AUTHORS,
        "description": " ".join(description.split()),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": version,
        "keywords": KEYWORDS + ["ERA5"],
        "related_identifiers": [
            {
                "identifier": ckpt["doi"],
                "relation": "requires",
                "resource_type": "other",
                "scheme": "doi",
            },
            _paper_relation(),
            {
                "identifier": REPO_URL,
                "relation": "isCompiledBy",
                "resource_type": "software",
                "scheme": "url",
            },
        ],
        "prereserve_doi": True,
    }


def _token(args):
    if args.token_file is not None:
        return args.token_file.expanduser().read_text().strip()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("give --token-file or set ZENODO_TOKEN")
    return token


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kind", required=True, choices=["checkpoint", "package"]
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="the sanitized checkpoint (checkpoint) or the package zip "
        "(package)",
    )
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "artifact"
        / "configs"
        / "model.yaml",
        help="model description, for the checkpoint record (default: "
        "%(default)s)",
    )
    parser.add_argument(
        "--version", default="1.0", help="record version (default: %(default)s)"
    )
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--deposition-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.kind == "checkpoint":
        model = yaml.safe_load(args.model_yaml.read_text())
        metadata = checkpoint_metadata(args.file, model, args.version)
    else:
        try:
            metadata = package_metadata(args.file, args.version)
        except ValueError as ex:
            sys.exit(str(ex))
    md5 = _md5(args.file)
    print(f"{args.file.name}: {args.file.stat().st_size} bytes, md5 {md5}")
    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("to review:\n- " + "\n- ".join(REVIEW_NOTES))
        return

    import requests

    try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    base = ZENODO_SANDBOX if args.sandbox else ZENODO
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {_token(args)}"

    if args.deposition_id is None:
        r = session.post(
            f"{base}/api/deposit/depositions", json={}, timeout=60
        )
    else:
        r = session.get(
            f"{base}/api/deposit/depositions/{args.deposition_id}", timeout=60
        )
    r.raise_for_status()
    deposition = r.json()
    if deposition.get("submitted"):
        sys.exit(f"deposition {deposition['id']} is already published")
    dep_id, bucket = deposition["id"], deposition["links"]["bucket"]

    # remove an earlier upload of the same file from the draft
    for f in deposition.get("files", []):
        if f["filename"] == args.file.name:
            session.delete(f["links"]["self"], timeout=60).raise_for_status()

    print(f"uploading {args.file.name} to draft {dep_id} ...")
    with open(args.file, "rb") as fh:
        r = session.put(f"{bucket}/{args.file.name}", data=fh, timeout=3600)
    r.raise_for_status()
    uploaded_md5 = r.json()["checksum"].removeprefix("md5:")
    if uploaded_md5 != md5:
        sys.exit(f"uploaded file md5 {uploaded_md5} != local {md5}")

    r = session.put(
        f"{base}/api/deposit/depositions/{dep_id}",
        json={"metadata": metadata},
        timeout=60,
    )
    r.raise_for_status()
    deposition = r.json()
    doi = deposition["metadata"].get("prereserve_doi", {}).get("doi")
    url = f"{base}/records/{dep_id}/files/{args.file.name}?download=1"
    print(f"draft {dep_id} ready for review: {deposition['links']['html']}")
    print(f"reserved DOI: {doi}")
    if args.kind == "checkpoint":
        print(
            "fill in artifact/configs/model.yaml (checkpoint):\n"
            f"  zenodo_record: {dep_id}\n  doi: {doi}\n  url: {url}\n"
            f"  md5: {md5}"
        )
    else:
        print(f"after publishing: PACKAGE_URL={url} PACKAGE_MD5={md5}")
    print("to review:\n- " + "\n- ".join(REVIEW_NOTES))
    print("NOT published: review and publish it in the Zenodo web UI")


if __name__ == "__main__":
    main()

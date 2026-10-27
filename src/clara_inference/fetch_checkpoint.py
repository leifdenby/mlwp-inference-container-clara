"""
Download the model checkpoint described in an inference artifact's
`configs/model.yaml` into the artifact directory, and check its md5.

Does nothing if the checkpoint is already there with the right md5. If the
download fails, the mirrors are tried in order.

The checkpoint of CLARA is a release asset of the private GitHub repository,
so downloads from GitHub use the token in the GITHUB_TOKEN environment
variable (in the container build: the `github_token` build secret). Private
release assets are downloaded through the API, from
`https://api.github.com/repos/<owner>/<repo>/releases/assets/<id>`.

Usage:
    python -m clara_inference.fetch_checkpoint <artifact_dir> \
        [--mirror-url <url>]...
"""
import argparse
import hashlib
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from loguru import logger


def _md5(fp):
    md5 = hashlib.md5()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            md5.update(chunk)
    return md5.hexdigest()


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """
    Don't forward the GitHub token to another host: GitHub redirects asset
    downloads to signed storage URLs, which reject an Authorization header.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and (
            urllib.parse.urlparse(newurl).hostname
            != urllib.parse.urlparse(req.full_url).hostname
        ):
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_DropAuthOnRedirect)


def _request(url):
    """The request for `url`, authenticated for GitHub with GITHUB_TOKEN."""
    headers = {}
    host = urllib.parse.urlparse(url).hostname or ""
    token = os.environ.get("GITHUB_TOKEN")
    if token and (host == "github.com" or host.endswith(".github.com")):
        headers["Authorization"] = f"Bearer {token}"
        if host == "api.github.com":
            # a release asset's content (not its JSON description)
            headers["Accept"] = "application/octet-stream"
    return urllib.request.Request(url, headers=headers)


def _download(url, fp, attempts=3, wait=10, timeout=60):
    """
    Download `url` to `fp`, retrying (with a growing wait) on network
    errors and server errors (e.g. a 504), but not client errors
    """
    for attempt in range(1, attempts + 1):
        try:
            with _opener.open(
                _request(url), timeout=timeout
            ) as response, open(fp, "wb") as fh:
                for chunk in iter(lambda: response.read(2**20), b""):
                    fh.write(chunk)
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as ex:
            client_error = (
                isinstance(ex, urllib.error.HTTPError) and ex.code < 500
            )
            if client_error or attempt == attempts:
                raise
            logger.warning(
                f"download failed ({ex}), retrying in {wait * attempt}s "
                f"(attempt {attempt}/{attempts})"
            )
            time.sleep(wait * attempt)


def fetch_checkpoint(artifact_dir, mirror_urls=()):
    artifact_dir = Path(artifact_dir)
    try:  # use the OS trust store (e.g. behind a TLS-inspecting proxy)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    model = yaml.safe_load(
        (artifact_dir / "configs" / "model.yaml").read_text()
    )
    ckpt = model["checkpoint"]
    fp = artifact_dir / ckpt["artifact_path"]
    if fp.exists() and _md5(fp) == ckpt["md5"]:
        logger.info(f"checkpoint already present and verified: {fp}")
        return fp

    fp_tmp = fp.with_name(fp.name + ".tmp")
    urls = [ckpt["url"], *mirror_urls]
    for i, url in enumerate(urls):
        logger.info(f"downloading {url} to {fp}")
        try:
            _download(url, fp_tmp)
            md5 = _md5(fp_tmp)
            if md5 != ckpt["md5"]:
                fp_tmp.unlink()
                raise ValueError(
                    f"md5 of downloaded checkpoint {md5} != {ckpt['md5']}"
                )
        except (OSError, http.client.HTTPException, ValueError) as ex:
            if i == len(urls) - 1:
                raise
            logger.warning(
                f"download from {url} failed ({ex}), trying the "
                f"next mirror"
            )
            continue
        fp_tmp.replace(fp)
        logger.info(f"checkpoint downloaded and verified: {fp}")
        return fp


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--mirror-url",
        action="append",
        default=[],
        help="a mirror of the checkpoint, tried (in order) if the download "
        "from model.yaml's url fails (repeatable)",
    )
    args = parser.parse_args()
    fetch_checkpoint(args.artifact_dir, mirror_urls=args.mirror_url)


if __name__ == "__main__":
    main()

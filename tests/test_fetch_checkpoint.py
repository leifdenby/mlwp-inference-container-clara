import hashlib
import io
import urllib.error

import pytest

from clara_inference import fetch_checkpoint


def _http_error(code):
    return urllib.error.HTTPError("url", code, "error", {}, None)


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(fetch_checkpoint.time, "sleep", lambda s: None)


def _urlopen(responses):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(response)

    return urlopen, calls


def test_download_retries_server_errors(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(504), b"checkpoint"])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    fp = tmp_path / "ckpt"
    fetch_checkpoint._download("url", fp)
    assert fp.read_bytes() == b"checkpoint"
    assert len(calls) == 2


def test_download_gives_up(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(504)] * 3)
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        fetch_checkpoint._download("url", tmp_path / "ckpt", attempts=3)
    assert len(calls) == 3


def test_download_no_retry_on_client_error(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(404), b"checkpoint"])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        fetch_checkpoint._download("url", tmp_path / "ckpt")
    assert len(calls) == 1


def _artifact(tmp_path, content):
    (tmp_path / "configs").mkdir()
    md5 = hashlib.md5(content).hexdigest()
    (tmp_path / "configs" / "model.yaml").write_text(
        "checkpoint:\n"
        "  url: https://zenodo/ckpt\n"
        f"  md5: {md5}\n"
        "  artifact_path: model.ckpt\n"
    )
    return tmp_path


def test_fetch_uses_mirror(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(504)] * 3 + [b"checkpoint"])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    artifact = _artifact(tmp_path, b"checkpoint")
    fp = fetch_checkpoint.fetch_checkpoint(
        artifact, mirror_urls=["https://mirror/ckpt"]
    )
    assert fp.read_bytes() == b"checkpoint"
    assert calls == ["https://zenodo/ckpt"] * 3 + ["https://mirror/ckpt"]


def test_fetch_md5_mismatch_uses_mirror(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([b"corrupt", b"checkpoint"])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    artifact = _artifact(tmp_path, b"checkpoint")
    fp = fetch_checkpoint.fetch_checkpoint(
        artifact, mirror_urls=["https://mirror/ckpt"]
    )
    assert fp.read_bytes() == b"checkpoint"
    assert len(calls) == 2


def test_fetch_all_fail(tmp_path, monkeypatch, no_wait):
    urlopen, calls = _urlopen([_http_error(404), _http_error(410)])
    monkeypatch.setattr(fetch_checkpoint.urllib.request, "urlopen", urlopen)
    artifact = _artifact(tmp_path, b"checkpoint")
    with pytest.raises(urllib.error.HTTPError) as ex:
        fetch_checkpoint.fetch_checkpoint(
            artifact, mirror_urls=["https://mirror/ckpt"]
        )
    assert ex.value.code == 410  # the last error
    assert not (artifact / "model.ckpt").exists()

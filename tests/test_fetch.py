import json

import pytest

from hkg_tmax import fetch
from hkg_tmax.fetch import FetchError, FetchPolicy, fetch_and_archive


class _FakeResponse:
    def __init__(self, *, status_code: int, content: bytes, url: str = "https://example.test/data"):
        self.status_code = status_code
        self.content = content
        self.url = url
        self.headers = {"content-type": "text/csv", "x-test": "present"}


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def get(self, url: str) -> _FakeResponse:
        return self.response


class _FakeSequenceClient:
    def __init__(self, items):
        self.items = items

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def get(self, url: str):
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_client(monkeypatch, response: _FakeResponse) -> None:
    monkeypatch.setattr(fetch.httpx, "Client", lambda *args, **kwargs: _FakeClient(response))


def _patch_client_sequence(monkeypatch, items) -> None:
    monkeypatch.setattr(fetch.httpx, "Client", lambda *args, **kwargs: _FakeSequenceClient(items))


def test_fetch_archives_http_metadata_and_hash(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse(status_code=200, content=b"a,b\n1,2\n"))

    snapshot = fetch_and_archive(
        url="https://example.test/data.csv",
        source_id="test_source",
        raw_root=tmp_path,
    )

    sidecar = json.loads(snapshot.sidecar_path.read_text())
    assert snapshot.content_path.read_bytes() == b"a,b\n1,2\n"
    assert sidecar["content_sha256"] == snapshot.sha256
    assert sidecar["content_length"] == 8
    assert sidecar["metadata"]["requested_url"] == "https://example.test/data.csv"
    assert sidecar["metadata"]["http_status"] == 200
    assert sidecar["metadata"]["response_headers"]["x-test"] == "present"


def test_fetch_rejects_http_error_without_archiving(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse(status_code=500, content=b"server error"))

    with pytest.raises(FetchError, match="HTTP 500"):
        fetch_and_archive(
            url="https://example.test/data.csv",
            source_id="test_source",
            raw_root=tmp_path,
        )

    assert list(tmp_path.rglob("*")) == []


def test_fetch_rejects_empty_payload_without_archiving(tmp_path, monkeypatch) -> None:
    _patch_client(monkeypatch, _FakeResponse(status_code=200, content=b""))

    with pytest.raises(FetchError, match="Empty payload"):
        fetch_and_archive(
            url="https://example.test/data.csv",
            source_id="test_source",
            raw_root=tmp_path,
        )

    assert list(tmp_path.rglob("*")) == []


def test_fetch_retries_transient_transport_error(tmp_path, monkeypatch) -> None:
    _patch_client_sequence(
        monkeypatch,
        [
            fetch.httpx.RemoteProtocolError("server disconnected"),
            _FakeResponse(status_code=200, content=b"a,b\n1,2\n"),
        ],
    )

    snapshot = fetch_and_archive(
        url="https://example.test/data.csv",
        source_id="test_source",
        raw_root=tmp_path,
        policy=FetchPolicy(max_attempts=2, retry_sleep_seconds=0),
    )

    assert snapshot.content_path.read_bytes() == b"a,b\n1,2\n"


def test_fetch_retries_exhaustion_without_archiving(tmp_path, monkeypatch) -> None:
    _patch_client_sequence(
        monkeypatch,
        [
            fetch.httpx.RemoteProtocolError("server disconnected"),
            fetch.httpx.RemoteProtocolError("server disconnected again"),
        ],
    )

    with pytest.raises(FetchError, match="after 2 attempt"):
        fetch_and_archive(
            url="https://example.test/data.csv",
            source_id="test_source",
            raw_root=tmp_path,
            policy=FetchPolicy(max_attempts=2, retry_sleep_seconds=0),
        )

    assert list(tmp_path.rglob("*")) == []

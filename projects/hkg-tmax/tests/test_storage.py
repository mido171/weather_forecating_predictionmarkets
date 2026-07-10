import json
from datetime import UTC, datetime

from hkg_tmax.hashing import sha256_bytes
from hkg_tmax.storage import store_raw_bytes


def test_raw_snapshot_is_hashed_and_has_sidecar(tmp_path) -> None:
    content = b"Year,Month,Day,Value\n2026,6,18,31.2\n"
    retrieved = datetime(2026, 6, 18, 12, tzinfo=UTC)
    snapshot = store_raw_bytes(
        tmp_path,
        source_id="test_source",
        content=content,
        retrieved_at=retrieved,
        extension="csv",
        metadata={"url": "https://example.invalid/data.csv"},
    )
    assert snapshot.content_path.read_bytes() == content
    assert snapshot.sha256 == sha256_bytes(content)
    sidecar = json.loads(snapshot.sidecar_path.read_text())
    assert sidecar["content_sha256"] == snapshot.sha256
    assert sidecar["content_length"] == len(content)
    assert sidecar["retrieved_at"] == "2026-06-18T12:00:00Z"
    assert sidecar["storage_schema_version"] == 2
    assert sidecar["content_relpath"].endswith("test_source/2026/06/18/" + snapshot.content_path.name)
    assert sidecar["sidecar_relpath"].endswith(
        "test_source/2026/06/18/" + snapshot.sidecar_path.name
    )
    assert sidecar["legacy_content_path"] == str(snapshot.content_path.resolve())


def test_identical_snapshot_same_timestamp_is_idempotent(tmp_path) -> None:
    kwargs = {
        "source_id": "test_source",
        "content": b"same",
        "retrieved_at": datetime(2026, 6, 18, 12, tzinfo=UTC),
        "extension": "txt",
    }
    first = store_raw_bytes(tmp_path, **kwargs)
    second = store_raw_bytes(tmp_path, **kwargs)
    assert first.content_path == second.content_path
    assert first.sidecar_path == second.sidecar_path


def test_long_source_id_does_not_make_temp_prefix_too_long(tmp_path) -> None:
    snapshot = store_raw_bytes(
        tmp_path,
        source_id="source_" + "x" * 80,
        content=b"payload",
        retrieved_at=datetime(2026, 6, 18, 12, tzinfo=UTC),
        extension="json",
    )

    assert snapshot.content_path.read_bytes() == b"payload"
    assert json.loads(snapshot.sidecar_path.read_text())["content_sha256"] == snapshot.sha256

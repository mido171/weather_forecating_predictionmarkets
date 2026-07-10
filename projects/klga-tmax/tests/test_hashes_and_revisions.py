from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from klga_tmax.ingestion.bronze import CurrentBronzeRecord, decide_bronze_revision
from klga_tmax.ingestion.hash_keys import canonical_json, payload_hash, source_request_id


def test_canonical_json_is_key_order_stable() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_source_request_id_uses_retrieval_bucket() -> None:
    first = source_request_id(
        source_name="iem_mos",
        source_endpoint="https://mesonet.agron.iastate.edu/mos/",
        request_params={"station": "LGA", "product": "MAV"},
        retrieved_at_utc=datetime(2026, 6, 27, 13, 0, 5, tzinfo=timezone.utc),
    )
    second = source_request_id(
        source_name="iem_mos",
        source_endpoint="https://mesonet.agron.iastate.edu/mos/",
        request_params={"product": "MAV", "station": "LGA"},
        retrieved_at_utc=datetime(2026, 6, 27, 13, 0, 55, tzinfo=timezone.utc),
    )
    assert first == second


def test_duplicate_bronze_payload_returns_existing_record_id() -> None:
    current_id = uuid4()
    current = CurrentBronzeRecord(
        source_record_id=current_id,
        payload_hash=payload_hash({"temp": 84}),
        revision_number=1,
    )
    decision = decide_bronze_revision(
        current_record=current,
        new_payload_hash=payload_hash({"temp": 84}),
    )
    assert decision.action == "return_existing"
    assert decision.source_record_id == current_id
    assert decision.revision_number == 1
    assert not decision.mark_prior_current_false


def test_changed_bronze_payload_creates_new_revision_and_supersedes_current() -> None:
    current_id = uuid4()
    current = CurrentBronzeRecord(
        source_record_id=current_id,
        payload_hash=payload_hash({"temp": 84}),
        revision_number=2,
    )
    decision = decide_bronze_revision(
        current_record=current,
        new_payload_hash=payload_hash({"temp": 85}),
    )
    assert decision.action == "insert_revision"
    assert decision.revision_number == 3
    assert decision.supersedes_source_record_id == current_id
    assert decision.mark_prior_current_false

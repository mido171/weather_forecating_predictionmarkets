from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class CurrentBronzeRecord:
    source_record_id: UUID
    payload_hash: str
    revision_number: int
    is_current: bool = True


@dataclass(frozen=True)
class BronzeRevisionDecision:
    action: str
    source_record_id: UUID | None
    revision_number: int
    supersedes_source_record_id: UUID | None
    mark_prior_current_false: bool


def decide_bronze_revision(
    *,
    current_record: CurrentBronzeRecord | None,
    new_payload_hash: str,
) -> BronzeRevisionDecision:
    if current_record is None:
        return BronzeRevisionDecision(
            action="insert_new",
            source_record_id=None,
            revision_number=1,
            supersedes_source_record_id=None,
            mark_prior_current_false=False,
        )

    if current_record.payload_hash == new_payload_hash:
        return BronzeRevisionDecision(
            action="return_existing",
            source_record_id=current_record.source_record_id,
            revision_number=current_record.revision_number,
            supersedes_source_record_id=None,
            mark_prior_current_false=False,
        )

    return BronzeRevisionDecision(
        action="insert_revision",
        source_record_id=None,
        revision_number=current_record.revision_number + 1,
        supersedes_source_record_id=current_record.source_record_id,
        mark_prior_current_false=True,
    )

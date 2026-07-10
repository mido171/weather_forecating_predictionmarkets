"""Model-selection artifact helpers and deterministic tie-breaking."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    expert_id: str
    validation_mae_c: float
    baseline_mae_c: float
    row_count: int
    complexity_rank: int

    @property
    def delta_mae_vs_baseline_c(self) -> float:
        return self.validation_mae_c - self.baseline_mae_c


@dataclass(frozen=True)
class ModelSelection:
    selected_candidate_id: str
    expert_id: str
    promoted: bool
    router_weight_cap: float
    validation_mae_c: float
    baseline_mae_c: float
    delta_mae_vs_baseline_c: float
    row_count: int
    tie_breaker: str

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def select_model_candidate(
    candidates: list[CandidateMetric],
    *,
    required_improvement_c: float,
    promoted_weight_cap: float,
    demoted_weight_cap: float = 0.0,
) -> ModelSelection:
    """Select the best candidate with stable tie-breaking and promotion status."""
    if not candidates:
        raise ValueError("At least one model candidate is required")
    selected = sorted(
        candidates,
        key=lambda item: (
            item.validation_mae_c,
            -item.row_count,
            item.complexity_rank,
            item.candidate_id,
        ),
    )[0]
    promoted = selected.validation_mae_c <= selected.baseline_mae_c - required_improvement_c
    return ModelSelection(
        selected_candidate_id=selected.candidate_id,
        expert_id=selected.expert_id,
        promoted=promoted,
        router_weight_cap=promoted_weight_cap if promoted else demoted_weight_cap,
        validation_mae_c=selected.validation_mae_c,
        baseline_mae_c=selected.baseline_mae_c,
        delta_mae_vs_baseline_c=selected.delta_mae_vs_baseline_c,
        row_count=selected.row_count,
        tie_breaker="mae,row_count_desc,complexity_rank,candidate_id",
    )

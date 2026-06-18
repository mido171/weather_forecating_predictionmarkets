from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .experiments import experiment_statuses


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", r"\|").replace("\n", " ").strip() or "—"


def _eligible(status: dict[str, Any]) -> bool:
    decision = status.get("decision") or {}
    gates = status.get("gates") or {}
    required = (
        "target_parity",
        "data_provenance",
        "asof_leakage",
        "reproducibility",
        "locked_oos",
        "calibration",
        "robustness",
        "operational_viability",
    )
    return (
        status.get("status") == "ACCEPTED"
        and decision.get("milestone_eligible") is True
        and all(gates.get(name) == "PASS" for name in required)
    )


def render_milestones(root: Path) -> Path:
    statuses = experiment_statuses(root)
    accepted = [status for status in statuses if _eligible(status)]
    rejected = [
        status
        for status in statuses
        if status.get("status") in {"REJECTED", "INCONCLUSIVE", "BLOCKED"}
    ]
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    lines = [
        "# HKG Tmax Milestones",
        "",
        f"**Last generated:** {timestamp}  ",
        "**Primary horizon:** read from `config/asof.yaml`  ",
        "**Production status:** disabled until `docs/07_PRODUCTION_GATE.md` passes",
        "",
        "## Current champion",
        "",
    ]

    if accepted:
        champion = accepted[-1]
        decision = champion.get("decision") or {}
        lines.append(
            f"Latest eligible accepted milestone: **{_cell(champion.get('experiment_id'))} — "
            f"{_cell(champion.get('title'))}**."
        )
        lines.append("")
        lines.append(_cell(decision.get("primary_conclusion")))
    else:
        lines.append(
            "No champion model is eligible yet. Any forecast generated before G1–G5 pass is exploratory only."
        )

    lines.extend(
        [
            "",
            "## Accepted milestone findings",
            "",
            "| Experiment | Finding | OOS delta | Primary metric | Sample |",
            "|---|---|---:|---|---:|",
        ]
    )
    if not accepted:
        lines.append("| — | No accepted findings yet | — | — | — |")
    else:
        for status in accepted:
            decision = status.get("decision") or {}
            metrics = status.get("_metrics") or {}
            directory = status.get("_directory")
            experiment_id = _cell(status.get("experiment_id"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"[{experiment_id}](experiments/{directory}/README.md)",
                        _cell(decision.get("primary_conclusion")),
                        _cell(decision.get("oos_delta")),
                        _cell(metrics.get("primary_metric")),
                        _cell(metrics.get("sample_size")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Baseline scoreboard",
            "",
            "| Model | Horizon | Sample | MAE °C | CRPS | Multiclass log loss | Calibration | Status |",
            "|---|---|---:|---:|---:|---:|---|---|",
            "| Seasonal climatology | TBD | — | — | — | — | — | Not run |",
            "| Persistence/anomaly | TBD | — | — | — | — | — | Not run |",
            "| HKO official forecast | TBD | — | — | — | — | — | Not archived |",
            "| Raw NWP consensus | TBD | — | — | — | — | — | Not archived |",
            "| Bias-corrected NWP | TBD | — | — | — | — | — | Not run |",
            "",
            "## Required gates",
            "",
            "- [ ] G0 environment and archival smoke test",
            "- [ ] G1 contract target and Daily Extract parity",
            "- [ ] G2 primary horizon selected and frozen",
            "- [ ] G3 source inventory and historical acquisition",
            "- [ ] G4 data quality and station-history audit",
            "- [ ] G5 strong baselines",
            "- [ ] G6 classical mechanism experiments",
            "- [ ] G7 expert probabilistic stack",
            "- [ ] G8 ML eligibility gate",
            "- [ ] G9 executable market evaluation",
            "- [ ] G10 production/shadow gate",
            "",
            "## Rejected, inconclusive, or blocked hypotheses",
            "",
            "| Experiment | Status | Conclusion |",
            "|---|---|---|",
        ]
    )
    if not rejected:
        lines.append("| — | — | None yet |")
    else:
        for status in rejected:
            directory = status.get("_directory")
            decision = status.get("decision") or {}
            experiment_id = _cell(status.get("experiment_id"))
            lines.append(
                f"| [{experiment_id}](experiments/{directory}/README.md) | "
                f"{_cell(status.get('status'))} | "
                f"{_cell(decision.get('primary_conclusion'))} |"
            )

    lines.extend(
        [
            "",
            "## Live blockers",
            "",
            "1. Historical-label parity must pass.",
            "2. Authentic point-in-time forecast-vintage coverage must be measured.",
            "3. The primary pre-event cutoff must be selected and frozen.",
            "4. Full historical order-book depth cannot be assumed; prospective archival is required.",
            "",
            "This file is generated from experiment gate status. Do not manually promote an unreviewed result.",
            "",
        ]
    )
    path = root / "MILESTONES.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

"""Build the bounded, metadata-only registry for top-level HKG scripts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PREFIX_CATEGORIES = {
    "audit_": "audit",
    "check_": "diagnostic",
    "inspect_": "diagnostic",
    "profile_": "diagnostic",
    "test_": "diagnostic",
    "build_": "builder",
    "generate_": "builder",
    "normalize_": "builder",
    "run_": "workflow",
    "backfill_": "acquisition",
    "fetch_": "acquisition",
    "download_": "acquisition",
    "install_": "operations",
    "start_": "operations",
    "monitor_": "operations",
    "resume_": "operations",
    "finalize_": "operations",
    "reset_": "operations",
}

# These classifications are deliberately explicit. The retained scripts back
# archived experiment evidence or compatibility imports; their presence in the
# flat directory does not make them current research authority.
LIFECYCLE_OVERRIDES = {
    "audit_hkg_t24_baseline_reproduction.py": "retained_reproduction",
    "backfill_hko_info_gov_hourly_readings.py": "active_operator",
    "build_hkg_tmax_phase_ab_analysis.py": "retained_reproduction",
    "export_hko_press_archive_offline.py": "active_operator",
    "finalize_hko_official_backfill.py": "active_operator",
    "load_datasets_to_mysql.py": "active_operator",
    "organize_human_dataset_folders.py": "active_operator",
    "profile_dataset_attributes_for_gpt_pro.py": "active_operator",
    "reset_tactical_gribstream_store.py": "active_operator",
    "resume_hko_official_backfill.py": "active_operator",
    "run_hkg_t24_0093_guarded_champion_sensitivity_check.py": "retained_reproduction",
    "run_hkg_t24_0095_mam_error_direction_split_lab.py": "retained_reproduction",
    "run_hkg_t24_0102_timestamp_proof_unlock_queue.py": "retained_reproduction",
    "run_hkg_t24_0105_0183_beastmode_roadmap.py": "retained_reproduction",
    "run_hkg_t24_0184_hf_teacher_proxy_causal_memory_router.py": "retained_reproduction",
    "run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py": "active_research",
    "run_hkg_t24_beastmode_signal_discovery.py": "retained_reproduction",
    "run_hkg_t24_candidate_timestamp_eligibility_audit.py": "retained_reproduction",
    "run_hkg_t24_gated_residual_specialist_screen.py": "retained_reproduction",
    "run_hkg_t24_long_history_cross_family_interaction_atlas.py": "retained_reproduction",
    "run_hkg_t24_r02_long_history.py": "retained_reproduction",
    "run_hkg_t24_r03_tmax_anatomy.py": "retained_reproduction",
    "run_hkg_t24_r04_thermal_trajectory.py": "retained_reproduction",
    "run_hkg_t24_r05_thermal_memory.py": "retained_reproduction",
    "run_hkg_t24_r06_moisture_state.py": "retained_reproduction",
    "run_hkg_t24_r07_transition_detection.py": "retained_reproduction",
    "run_hkg_t24_r08_wind_advection.py": "retained_reproduction",
    "run_hkg_t24_r09_station_temperature_gradient.py": "retained_reproduction",
    "run_hkg_t24_r10_latent_spatial_modes.py": "retained_reproduction",
    "run_hkg_t24_r11_dynamic_upwind_station_selection.py": "retained_reproduction",
    "run_hkg_t24_r12_solar_radiation.py": "retained_reproduction",
    "run_hkg_t24_r13_to_r30_precondition_gates.py": "retained_reproduction",
    "run_hkg_t24_r14_r17_robust_long_history.py": "retained_reproduction",
    "run_hkg_t24_station_contribution_atlas.py": "retained_reproduction",
    "run_hkg_tmax_baselines.py": "retained_reproduction",
    "run_hko_official_remaining_backfill.py": "active_operator",
}


def classify(name: str) -> str:
    for prefix, category in PREFIX_CATEGORIES.items():
        if name.startswith(prefix):
            return category
    return "utility"


def risk_for(category: str, text: str) -> str:
    if category == "acquisition" or "requests." in text or "httpx." in text:
        return "network"
    if category == "operations" or "connect_db(" in text or "create_engine(" in text:
        return "mutation_or_process"
    return "offline_or_derived"


def lifecycle_for(name: str, category: str) -> str:
    override = LIFECYCLE_OVERRIDES.get(name)
    if override is not None:
        return override
    if category in {"acquisition", "operations"}:
        return "operator_workflow"
    if category == "workflow":
        return "research_workflow"
    return "maintained_utility"


def build_registry(scripts_root: Path, output: Path) -> int:
    rows: list[dict[str, object]] = []
    for path in sorted(scripts_root.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(512 * 1024)
        category = classify(path.name)
        rows.append(
            {
                "script": path.name,
                "category": category,
                "lifecycle": lifecycle_for(path.name, category),
                "risk": risk_for(category, text),
                "project_paths_detected": "ProjectPaths" in text,
                "execute_guard_detected": "--execute" in text,
                "main_guard_detected": 'if __name__ == "__main__"' in text,
                "bytes": path.stat().st_size,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ())
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("scripts/REGISTRY.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    count = build_registry(root, args.output.resolve())
    print(f"Registered {count} scripts in {args.output}")


if __name__ == "__main__":
    main()

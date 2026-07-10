from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

import hkg_tmax_probability.reporting as reporting
import scripts.run_hkg_tmax_0004_station_hour_residual_information_atlas as atlas
import scripts.run_hkg_tmax_probability_distribution_methods_v2 as distribution_v2
from hkg_tmax.evaluation.reporting import (
    demote_markdown_headings,
    write_bounded_readme_section,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PRODUCER_SCRIPTS = (
    "audit_tactical_gribstream_deep_sanity.py",
    "backfill_public_weather_to_postgres.py",
    "benchmark_public_daily_coverage_gfs_gefs_himawari.py",
    "benchmark_public_weather_speed_optimization.py",
    "build_last2_gfs_gefs_radar_structured_delivery.py",
    "decode_himawari_hko_b13_item.py",
    "fetch_public_gfs_gefs_himawari_smoke.py",
    "normalize_public_gfs_gefs_himawari_smoke.py",
    "reset_tactical_gribstream_store.py",
    "run_hkg_t24_0215_gpt_pro_point_forecast_strategy.py",
    "run_hkg_tmax_0004_station_hour_residual_information_atlas.py",
    "run_hkg_tmax_probability_bucket_v1.py",
    "run_hkg_tmax_probability_distribution_methods_v2.py",
    "run_hkg_tmax_residual_ml_next_round.py",
    "run_hkg_tmax_residual_ml_official_memory.py",
    "run_hkg_tmax_residual_ml_strategy.py",
    "run_public_gfs_gefs_himawari_7day_backfill_rehearsal.py",
    "run_public_weather_backfill_day_shards.py",
    "run_tactical_gribstream_batch_smoke.py",
    "run_tactical_gribstream_first_week.py",
    "run_tactical_gribstream_h24n_smoke.py",
)


def _leaderboard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "method": "challenger",
                "family": "v2",
                "rps": 0.08,
                "relative_rps_gain_vs_b4": 0.02,
                "fold14_relative_rps_gain_vs_b4": 0.01,
                "presealed_relative_rps_gain_vs_b4": 0.00,
                "nll": 1.20,
                "brier": 0.10,
                "ece": 0.03,
                "gates": "fail:presealed_rps_gain",
                "champion_flag": False,
            },
            {
                "rank": 2,
                "method": "B4_hierarchical_residual_pmf",
                "family": "baseline",
                "rps": 0.09,
                "relative_rps_gain_vs_b4": 0.00,
                "fold14_relative_rps_gain_vs_b4": 0.00,
                "presealed_relative_rps_gain_vs_b4": 0.00,
                "nll": 1.18,
                "brier": 0.09,
                "ece": 0.02,
                "gates": "baseline_retained",
                "champion_flag": True,
            },
        ]
    )


def test_atlas_writes_one_idempotent_readme_and_machine_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(atlas, "EXPERIMENT_DIR", tmp_path)
    (tmp_path / "README.md").write_text(
        "# Curated Atlas Dossier\n\nKeep this manual context.\n",
        encoding="utf-8",
    )
    summary = pd.DataFrame(
        [
            {
                "feature_name": "hko__latest_temp_c",
                "feature_family": "hko_latest",
                "station": None,
                "station_role": "hko",
                "transform": "latest_temp",
                "window_hours": None,
                "snapshot_hour": None,
                "n": 1000,
                "max_abs_primary_corr": 0.2,
                "pearson_residual": 0.2,
                "pearson_abs_error": 0.1,
                "pearson_under_gt1": 0.1,
                "pearson_over_gt1": -0.1,
                "pearson_hot_under": 0.1,
                "residual_corr_train_eval_same_sign": True,
            }
        ]
    )
    metrics = {
        "generated_at_utc": "2026-07-10T00:00:00Z",
        "significance": {
            "significance_score_1_to_100": 73,
            "best_actionability": None,
        },
    }
    db_counts = {
        "frame_rows": "1000",
        "frame_min_date": "2000-01-02",
        "frame_max_date": "2023-12-31",
        "hourly_rows_24h_join": "12000",
        "station_long_rows_24h_join": "24000",
        "feature_value_rows": "48000",
        "feature_count": "48",
        "station_count": "12",
        "uses_confirmation_rows": "0",
    }

    atlas.write_static_protocol_artifacts(
        "postgresql://research_user:placeholder-password@127.0.0.1:5432/hkg_tmax_research"
    )
    atlas.write_experiment_readme(
        summary=summary,
        spearman=pd.DataFrame(),
        actionability=pd.DataFrame(),
        station_board=pd.DataFrame(),
        family_board=pd.DataFrame(),
        metrics=metrics,
        db_counts=db_counts,
    )
    first_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    atlas.write_experiment_readme(
        summary=summary,
        spearman=pd.DataFrame(),
        actionability=pd.DataFrame(),
        station_board=pd.DataFrame(),
        family_board=pd.DataFrame(),
        metrics=metrics,
        db_counts=db_counts,
    )

    assert list(tmp_path.rglob("*.md")) == [tmp_path / "README.md"]
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == first_readme
    assert first_readme.startswith("# Curated Atlas Dossier\n\nKeep this manual context.")
    assert first_readme.count(atlas.README_GENERATED_START) == 1
    assert first_readme.count(atlas.README_GENERATED_END) == 1
    assert "## Hypothesis" in first_readme
    assert "## As-Of Contract" in first_readme
    assert "## Protocol" in first_readme
    assert "## Results" in first_readme
    assert "## Conclusion" in first_readme
    assert "## Reproduce" in first_readme
    assert "## Evidence Map" in first_readme
    assert (tmp_path / "STATUS.yaml").is_file()
    assert (tmp_path / "RUN_CONFIG.yaml").is_file()
    manifest_text = (tmp_path / "DATA_MANIFEST.yaml").read_text(encoding="utf-8")
    assert "research_user:***@" in manifest_text
    assert "placeholder-password" not in manifest_text


def test_probability_v1_writes_json_selection_summary_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = _leaderboard()
    reporting.write_model_summary(
        tmp_path,
        leaderboard,
        leakage={"status": "pass", "total_violations": 0},
        label_audit={"status": "pass", "bucket_changes": 0},
        acceptance_gates={"nll_worse_than_b4_max": 0.005},
    )
    monkeypatch.setattr(reporting.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")
    reporting.write_manifest(
        tmp_path,
        tmp_path / "config.yaml",
        [path.name for path in tmp_path.iterdir()],
    )

    summary = json.loads((tmp_path / "model_selection_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert summary["selection"]["champion"]["method"] == "B4_hierarchical_residual_pmf"
    assert summary["selection"]["champion"]["rps"] == 0.09
    assert summary["selection"]["promotion_gates"] == {"nll_worse_than_b4_max": 0.005}
    assert summary["audits"]["leakage"]["status"] == "pass"
    assert summary["metric_artifacts"]["by_split"] == "scoreboard_by_split.csv"
    assert {artifact["path"] for artifact in manifest["artifacts"]} == {
        "model_selection_summary.json"
    }
    assert not list(tmp_path.glob("*.md"))


def test_atlas_rejects_malformed_generated_readme_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(atlas.README_GENERATED_START, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Malformed generated README markers"):
        atlas.write_bounded_readme_section(readme, "## Generated")


def test_probability_v2_writes_one_machine_readable_selection_summary_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    leaderboard = _leaderboard()
    fold14 = leaderboard.assign(split="fold_1_4")
    presealed = leaderboard.assign(split="presealed_2022_2023")
    distribution_v2._write_model_selection_summary(
        tmp_path,
        leaderboard,
        fold14,
        presealed,
        {"acceptance_gates": {"complex_vs_b4_presealed_min_rps_gain": 0.01}},
        {"status": "pass", "total_violations": 0},
        {"status": "pass", "bucket_changes": 0},
        {"status": "pass", "violations": 0},
    )
    monkeypatch.setattr(
        distribution_v2.subprocess,
        "check_output",
        lambda *args, **kwargs: "abc123\n",
    )
    distribution_v2._write_manifest(
        tmp_path,
        tmp_path / "config.yaml",
        [path.name for path in tmp_path.iterdir()],
    )

    summary = json.loads((tmp_path / "model_selection_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert summary["selection"]["champion"]["method"] == "B4_hierarchical_residual_pmf"
    assert summary["selection"]["raw_lowest_rps"]["method"] == "challenger"
    assert summary["selection"]["promotion_gates"] == {"complex_vs_b4_presealed_min_rps_gain": 0.01}
    assert len(summary["scoreboard_extracts"]["fold_1_4_top_20"]) == 2
    assert {artifact["path"] for artifact in manifest["artifacts"]} == {
        "model_selection_summary.json"
    }
    assert not list(tmp_path.glob("*.md"))


def test_shared_readme_writer_is_bounded_idempotent_and_fail_closed(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Curated dossier\n\nManual evidence.\n", encoding="utf-8")

    kwargs = {
        "start_marker": "<!-- BEGIN GENERATED TEST -->",
        "end_marker": "<!-- END GENERATED TEST -->",
        "section": "## Latest result\n\nPassed.",
        "default_title": "Fallback",
    }
    write_bounded_readme_section(readme, **kwargs)
    first = readme.read_text(encoding="utf-8")
    write_bounded_readme_section(readme, **kwargs)

    assert readme.read_text(encoding="utf-8") == first
    assert first.startswith("# Curated dossier\n\nManual evidence.")
    assert first.count(kwargs["start_marker"]) == 1
    assert first.count(kwargs["end_marker"]) == 1

    reversed_markers = tmp_path / "reversed" / "README.md"
    reversed_markers.parent.mkdir()
    reversed_markers.write_text(
        f"{kwargs['end_marker']}\ntext\n{kwargs['start_marker']}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Malformed generated README markers"):
        write_bounded_readme_section(reversed_markers, **kwargs)

    with pytest.raises(ValueError, match="must target README.md"):
        write_bounded_readme_section(tmp_path / "RESULTS.md", **kwargs)


def test_heading_demotion_preserves_fenced_code() -> None:
    source = "# Result\n\n## Detail\n\n```markdown\n# literal\n```\n"

    assert demote_markdown_headings(source) == (
        "## Result\n\n### Detail\n\n```markdown\n# literal\n```"
    )


def test_campaign_producers_do_not_write_noncanonical_markdown_literals() -> None:
    write_functions = {
        "write_text",
        "write_bounded_readme_section",
        "write_report",
    }
    violations: list[str] = []
    for script_name in CAMPAIGN_PRODUCER_SCRIPTS:
        path = PROJECT_ROOT / "scripts" / script_name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name not in write_functions:
                continue
            markdown_literals = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value.lower().endswith(".md")
            }
            for literal in markdown_literals:
                if Path(literal).name.casefold() != "readme.md":
                    violations.append(f"{script_name}:{node.lineno}:{literal}")

    assert violations == []

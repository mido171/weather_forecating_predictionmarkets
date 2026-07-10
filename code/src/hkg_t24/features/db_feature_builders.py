"""Database-backed Jira 002 feature-family builders."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from hkg_t24.artifacts.reports import ReportWriter
from hkg_t24.constants import (
    CUTOFF_ID,
    PROXY_SCHEMA_VERSION,
    STRICT_SCHEMA_VERSION,
)
from hkg_t24.features.diagnostic_proxy import lagged_climate_proxy_features
from hkg_t24.features.feature_dictionary import write_feature_dictionaries
from hkg_t24.features.matrix_builder import (
    FeatureValue,
    build_scoped_matrix_rows,
    build_strict_matrix_rows,
    persist_feature_matrix_rows,
)
from hkg_t24.features.nwp_daily import (
    HKG_NWP_LOCATIONS,
    build_gefs_ensemble_features,
    build_gefs_mean_features,
    build_gfs_daily_features,
    shadow_center_feature_names,
)
from hkg_t24.features.official_anchor import OfficialForecastRow, official_feature_map
from hkg_t24.features.online_state import ResidualObservation, build_online_state
from hkg_t24.features.station_proxy import station_proxy_features
from hkg_t24.features.target_memory import build_target_memory_features
from hkg_t24.timeutils import iter_target_dates, snapshot_id


@dataclass(frozen=True)
class FeatureBuildSummary:
    scope: str
    feature_matrix_rows: int
    official_rows: int
    target_memory_rows: int
    online_state_rows: int
    nwp_feature_rows: int
    proxy_rows: int
    shadow_rows: int


def _regclass_exists(connection: Any, qualified_name: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (qualified_name,))
        row = cursor.fetchone()
    return row is not None and row[0] is not None


def _load_labels(connection: Any, *, start_date: date, end_date: date) -> dict[date, float]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT target_date_hkt, target_tmax_c::double precision
            FROM model_core.target_label
            WHERE target_date_hkt BETWEEN %s AND %s
              AND target_tmax_c IS NOT NULL
            ORDER BY target_date_hkt
            """,
            (start_date, end_date),
        )
        return {row[0]: float(row[1]) for row in cursor.fetchall()}


def _load_labels_for_memory(
    connection: Any,
    *,
    end_date: date,
) -> tuple[list[tuple[date, float | None]], str]:
    if _regclass_exists(connection, "feature_safe.hko_target_history_pre2024"):
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT local_date, target_tmax_c::double precision
                FROM feature_safe.hko_target_history_pre2024
                WHERE local_date <= %s
                  AND target_tmax_c IS NOT NULL
                ORDER BY local_date
                """,
                (end_date,),
            )
            rows = cursor.fetchall()
        return [(row[0], None if row[1] is None else float(row[1])) for row in rows], (
            "feature_safe.hko_target_history_pre2024"
        )
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT target_date_hkt, target_tmax_c::double precision
            FROM model_core.target_label
            WHERE target_date_hkt <= %s
            ORDER BY target_date_hkt
            """,
            (end_date,),
        )
        rows = cursor.fetchall()
    return [(row[0], None if row[1] is None else float(row[1])) for row in rows], "model_core.target_label"


def _persist_target_memory(
    connection: Any,
    features_by_date: Mapping[date, Mapping[str, FeatureValue]],
) -> None:
    with connection.cursor() as cursor:
        for target_date_hkt, features in features_by_date.items():
            cursor.execute(
                """
                INSERT INTO model_features.target_memory_features (
                  target_date_hkt, cutoff_id, snapshot_id, feature_schema_version,
                  features_jsonb, feature_count, leakage_status
                )
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,'passed')
                ON CONFLICT (target_date_hkt, cutoff_id, feature_schema_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  features_jsonb = EXCLUDED.features_jsonb,
                  feature_count = EXCLUDED.feature_count,
                  leakage_status = EXCLUDED.leakage_status,
                  generated_at_utc = now()
                """,
                (
                    target_date_hkt,
                    CUTOFF_ID,
                    snapshot_id(target_date_hkt),
                    STRICT_SCHEMA_VERSION,
                    json.dumps(features, sort_keys=True),
                    len(features),
                ),
            )


def build_target_memory_family(
    connection: Any,
    writer: ReportWriter,
    *,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, FeatureValue]]:
    labels, source_table = _load_labels_for_memory(connection, end_date=end_date)
    requested_dates = set(iter_target_dates(start_date, end_date))
    all_features = build_target_memory_features(labels, selected_dates=requested_dates)
    selected: dict[date, dict[str, FeatureValue]] = {
        target_date_hkt: {name: value for name, value in features.items()}
        for target_date_hkt, features in all_features.items()
    }
    _persist_target_memory(connection, selected)
    first_label = labels[0][0] if labels else None
    last_label = labels[-1][0] if labels else None
    pre_start_count = sum(1 for label_date, _ in labels if label_date < start_date)
    lag365_non_null = sum(1 for features in selected.values() if features.get("target__lag365_tmax_c") is not None)
    clim30_non_null = sum(1 for features in selected.values() if features.get("target__clim30_mean_c") is not None)
    writer.write_csv(
        "target_memory_history_coverage.csv",
        (
            "source_table",
            "source_label_count",
            "source_first_date",
            "source_last_date",
            "requested_start_date",
            "requested_end_date",
            "pre_start_label_count",
            "persisted_feature_rows",
            "lag365_non_null_rows",
            "clim30_non_null_rows",
            "h24n_finalized_label_rule",
        ),
        [
            (
                source_table,
                len(labels),
                first_label,
                last_label,
                start_date,
                end_date,
                pre_start_count,
                len(selected),
                lag365_non_null,
                clim30_non_null,
                "T-2 finalized target labels only; lag1 names are forbidden.",
            )
        ],
    )
    writer.write_root_report(
        "target_memory_history_coverage_report.md",
        "HKG-T24 Target-Memory History Coverage Report",
        (
            ("Status", "PASS" if first_label is not None and first_label <= date(1884, 1, 1) else "WARNING"),
            ("Source Table", f"`{source_table}`"),
            ("Source Date Range", f"{first_label}..{last_label} across {len(labels)} labels."),
            ("Requested Matrix Window", f"{start_date}..{end_date}."),
            ("Pre-Start History Rows", str(pre_start_count)),
            ("Persisted Feature Rows", str(len(selected))),
            ("Non-Null Long-History Features", f"lag365={lag365_non_null}; clim30={clim30_non_null}."),
            ("Safety Rule", "All finalized target-memory features remain bounded to T-2 or older."),
        ),
    )
    return selected


def _official_rows_by_date(
    connection: Any,
    *,
    start_date: date,
    end_date: date,
) -> dict[date, tuple[datetime, list[OfficialForecastRow]]]:
    grouped: dict[date, tuple[datetime, list[OfficialForecastRow]]] = {}
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
              cc.target_date_hkt,
              cc.operational_freeze_utc,
              h.issue_at_utc,
              h.target_issue_lead_days,
              h.forecast_min_c,
              h.forecast_max_c,
              coalesce(h.temperature_text, h.full_text) AS forecast_text,
              h.row_quality_status
            FROM model_core.cutoff_calendar cc
            LEFT JOIN public.hko_historical_forecasts_2000_2026 h
              ON h.target_date = cc.target_date_hkt
             AND h.issue_at_utc <= cc.operational_freeze_utc
             AND h.usable_local_tmax_forecast IS TRUE
             AND h.forecast_max_c IS NOT NULL
            WHERE cc.target_date_hkt BETWEEN %s AND %s
              AND cc.cutoff_id = 'H24N'
            ORDER BY cc.target_date_hkt, h.issue_at_utc
            """,
            (start_date, end_date),
        )
        for row in cursor.fetchall():
            target_date_hkt = row[0]
            freeze = row[1]
            current = grouped.setdefault(target_date_hkt, (freeze, []))
            if row[2] is None:
                continue
            current[1].append(
                OfficialForecastRow(
                    issue_at_utc=row[2],
                    forecast_min_c=None if row[4] is None else float(row[4]),
                    forecast_max_c=None if row[5] is None else float(row[5]),
                    forecast_text=None if row[6] is None else str(row[6]),
                    row_quality_status=str(row[7]),
                )
            )
    return grouped


def _official_anchor_diagnostics(
    connection: Any,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH joined AS (
              SELECT
                cc.target_date_hkt,
                cc.operational_freeze_utc,
                h.issue_at_utc,
                h.forecast_max_c,
                h.usable_local_tmax_forecast,
                h.target_issue_lead_days,
                h.row_quality_status
              FROM model_core.cutoff_calendar cc
              LEFT JOIN public.hko_historical_forecasts_2000_2026 h
                ON h.target_date = cc.target_date_hkt
               AND h.usable_local_tmax_forecast IS TRUE
               AND h.forecast_max_c IS NOT NULL
              WHERE cc.cutoff_id = 'H24N'
                AND cc.target_date_hkt BETWEEN %s AND %s
            )
            SELECT
              count(DISTINCT target_date_hkt),
              count(*) FILTER (WHERE usable_local_tmax_forecast IS TRUE),
              count(DISTINCT target_date_hkt) FILTER (WHERE usable_local_tmax_forecast IS TRUE),
              count(*) FILTER (WHERE usable_local_tmax_forecast IS TRUE AND issue_at_utc <= operational_freeze_utc),
              count(DISTINCT target_date_hkt) FILTER (
                WHERE usable_local_tmax_forecast IS TRUE AND issue_at_utc <= operational_freeze_utc
              ),
              count(DISTINCT target_date_hkt) FILTER (
                WHERE usable_local_tmax_forecast IS TRUE AND target_issue_lead_days >= 1
              ),
              min(target_date_hkt) FILTER (WHERE usable_local_tmax_forecast IS TRUE),
              max(target_date_hkt) FILTER (WHERE usable_local_tmax_forecast IS TRUE),
              min(extract(epoch from (issue_at_utc - operational_freeze_utc)) / 3600.0) FILTER (
                WHERE usable_local_tmax_forecast IS TRUE
              )
            FROM joined
            """,
            (start_date, end_date),
        )
        row = cursor.fetchone()
    return {
        "requested_dates": row[0],
        "usable_source_rows": row[1],
        "usable_source_dates": row[2],
        "strict_eligible_rows": row[3],
        "strict_eligible_dates": row[4],
        "diagnostic_lead1_dates": row[5],
        "usable_first_date": row[6],
        "usable_last_date": row[7],
        "minimum_issue_minus_freeze_hours": None if row[8] is None else float(row[8]),
    }


def _write_official_anchor_reports(
    connection: Any,
    writer: ReportWriter,
    *,
    start_date: date,
    end_date: date,
    built_rows: int,
    strict_non_null_rows: int,
) -> None:
    diagnostics = _official_anchor_diagnostics(connection, start_date=start_date, end_date=end_date)
    strict_dates_raw = diagnostics["strict_eligible_dates"]
    strict_dates = strict_dates_raw if isinstance(strict_dates_raw, int) else 0
    status = "PASS" if strict_dates > 0 else "BLOCKED_STRICT_E0"
    writer.write_csv(
        "official_anchor_eligibility_report.csv",
        (
            "requested_dates",
            "usable_source_rows",
            "usable_source_dates",
            "strict_eligible_rows",
            "strict_eligible_dates",
            "diagnostic_lead1_dates",
            "usable_first_date",
            "usable_last_date",
            "minimum_issue_minus_freeze_hours",
            "persisted_official_feature_rows",
            "strict_non_null_forecast_max_rows",
            "status",
        ),
        [
            (
                diagnostics["requested_dates"],
                diagnostics["usable_source_rows"],
                diagnostics["usable_source_dates"],
                diagnostics["strict_eligible_rows"],
                diagnostics["strict_eligible_dates"],
                diagnostics["diagnostic_lead1_dates"],
                diagnostics["usable_first_date"],
                diagnostics["usable_last_date"],
                diagnostics["minimum_issue_minus_freeze_hours"],
                built_rows,
                strict_non_null_rows,
                status,
            )
        ],
    )
    writer.write_root_report(
        "official_anchor_eligibility_report.md",
        "HKG-T24 Strict Official Anchor Eligibility Report",
        (
            ("Status", status),
            ("Strict H24N Rule", "Official rows must have `issue_at_utc <= operational_freeze_utc`."),
            (
                "Raw Source Coverage",
                f"usable_rows={diagnostics['usable_source_rows']}; usable_dates={diagnostics['usable_source_dates']}; "
                f"date_range={diagnostics['usable_first_date']}..{diagnostics['usable_last_date']}.",
            ),
            (
                "Strict Eligibility",
                f"eligible_rows={diagnostics['strict_eligible_rows']}; eligible_dates={diagnostics['strict_eligible_dates']}; "
                f"minimum_issue_minus_freeze_hours={diagnostics['minimum_issue_minus_freeze_hours']}.",
            ),
            (
                "Persisted Strict Features",
                f"official_feature_rows={built_rows}; non_null_official__forecast_max_c={strict_non_null_rows}.",
            ),
            (
                "Blocker",
                "Strict E0 remains unavailable until a source row is proven available by the operational freeze."
                if strict_dates == 0
                else "At least one strict official anchor date is available.",
            ),
        ),
    )

    with connection.cursor() as cursor:
        cursor.execute(
            """
            WITH ranked AS (
              SELECT
                h.target_date,
                h.forecast_max_c,
                row_number() OVER (
                  PARTITION BY h.target_date
                  ORDER BY h.issue_at_utc DESC NULLS LAST
                ) AS rn
              FROM public.hko_historical_forecasts_2000_2026 h
              WHERE h.target_date BETWEEN %s AND %s
                AND h.usable_local_tmax_forecast IS TRUE
                AND h.forecast_max_c IS NOT NULL
                AND h.target_issue_lead_days >= 1
            )
            SELECT
              count(*),
              min(r.target_date),
              max(r.target_date),
              avg(abs(r.forecast_max_c - l.target_tmax_c::double precision)),
              avg(r.forecast_max_c - l.target_tmax_c::double precision)
            FROM ranked r
            JOIN model_core.target_label l
              ON l.target_date_hkt = r.target_date
            WHERE r.rn = 1
            """,
            (start_date, end_date),
        )
        row = cursor.fetchone()
    writer.write_csv(
        "post_official_diagnostic_scoreboard.csv",
        ("scope", "row_count", "first_date", "last_date", "mae_c", "bias_forecast_minus_actual_c"),
        [
            (
                "diagnostic_post_official_lead1_not_strict_h24n",
                row[0],
                row[1],
                row[2],
                None if row[3] is None else float(row[3]),
                None if row[4] is None else float(row[4]),
            )
        ],
    )
    writer.write_root_report(
        "post_official_diagnostic_report.md",
        "HKG-T24 Post-Official Diagnostic Anchor Report",
        (
            ("Status", "DIAGNOSTIC_ONLY"),
            (
                "Reason",
                "This report scores the latest lead-1 official raw forecast without treating it as strict H24N input.",
            ),
            (
                "Metrics",
                f"rows={row[0]}; range={row[1]}..{row[2]}; "
                f"MAE={None if row[3] is None else round(float(row[3]), 6)}; "
                f"bias_forecast_minus_actual={None if row[4] is None else round(float(row[4]), 6)}.",
            ),
            ("Strict Impact", "Zero. These rows are not written into strict E0 unless pre-freeze eligibility is proven."),
        ),
    )


def build_official_family(
    connection: Any,
    writer: ReportWriter,
    *,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, FeatureValue]]:
    grouped = _official_rows_by_date(connection, start_date=start_date, end_date=end_date)
    output: dict[date, dict[str, FeatureValue]] = {}
    with connection.cursor() as cursor:
        for target_date_hkt, (freeze, rows) in grouped.items():
            features = official_feature_map(rows, operational_freeze_utc=freeze)
            output[target_date_hkt] = {name: value for name, value in features.items()}
            selected_issue = rows[-1].issue_at_utc if rows else None
            cursor.execute(
                """
                INSERT INTO model_features.official_features (
                  target_date_hkt, cutoff_id, snapshot_id, feature_schema_version,
                  official__forecast_min_c, official__forecast_max_c,
                  official__forecast_range_c, official__forecast_midpoint_c,
                  features_jsonb, selected_issue_at_utc, source_row_count, leakage_status
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,'passed')
                ON CONFLICT (target_date_hkt, cutoff_id, feature_schema_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  official__forecast_min_c = EXCLUDED.official__forecast_min_c,
                  official__forecast_max_c = EXCLUDED.official__forecast_max_c,
                  official__forecast_range_c = EXCLUDED.official__forecast_range_c,
                  official__forecast_midpoint_c = EXCLUDED.official__forecast_midpoint_c,
                  features_jsonb = EXCLUDED.features_jsonb,
                  selected_issue_at_utc = EXCLUDED.selected_issue_at_utc,
                  source_row_count = EXCLUDED.source_row_count,
                  leakage_status = EXCLUDED.leakage_status,
                  generated_at_utc = now()
                """,
                (
                    target_date_hkt,
                    CUTOFF_ID,
                    snapshot_id(target_date_hkt),
                    STRICT_SCHEMA_VERSION,
                    features.get("official__forecast_min_c"),
                    features.get("official__forecast_max_c"),
                    features.get("official__forecast_range_c"),
                    features.get("official__forecast_midpoint_c"),
                    json.dumps(features, sort_keys=True),
                    selected_issue,
                    len(rows),
                ),
            )
            revision_features = {
                name: features[name]
                for name in (
                    "official__revision_count",
                    "official__revision_min_delta_c",
                    "official__revision_max_delta_c",
                    "official__revision_range_delta_c",
                )
            }
            cursor.execute(
                """
                INSERT INTO model_features.official_revision_features (
                  target_date_hkt, cutoff_id, snapshot_id, feature_schema_version,
                  revision_count, features_jsonb, leakage_status
                )
                VALUES (%s,%s,%s,%s,%s,%s::jsonb,'passed')
                ON CONFLICT (target_date_hkt, cutoff_id, feature_schema_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  revision_count = EXCLUDED.revision_count,
                  features_jsonb = EXCLUDED.features_jsonb,
                  leakage_status = EXCLUDED.leakage_status,
                  generated_at_utc = now()
                """,
                (
                    target_date_hkt,
                    CUTOFF_ID,
                    snapshot_id(target_date_hkt),
                    STRICT_SCHEMA_VERSION,
                    len(rows),
                    json.dumps(revision_features, sort_keys=True),
                ),
            )
    strict_non_null_rows = sum(1 for features in output.values() if features.get("official__forecast_max_c") is not None)
    _write_official_anchor_reports(
        connection,
        writer,
        start_date=start_date,
        end_date=end_date,
        built_rows=len(output),
        strict_non_null_rows=strict_non_null_rows,
    )
    writer.write_root_report(
        "official_anchor_coverage.md",
        "HKG-T24-002 Official Anchor Coverage",
        (
            ("Status", "PASS" if strict_non_null_rows else "BLOCKED_STRICT_E0"),
            ("Rows", f"Official feature rows built: {len(output)}; strict non-null max rows: {strict_non_null_rows}."),
            (
                "Selection Rule",
                "Latest usable official forecast row with `issue_at_utc <= operational_freeze_utc`; "
                "post-freeze official forecasts remain diagnostic-only.",
            ),
        ),
    )
    return output


def _load_residual_observations(connection: Any, *, end_date: date) -> list[ResidualObservation]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.target_date_hkt, p.expert_id, p.prediction_tmax_c, l.target_tmax_c::double precision
            FROM model_oof.expert_prediction p
            JOIN model_core.target_label l
              ON l.target_date_hkt = p.target_date_hkt
            WHERE p.prediction_tmax_c IS NOT NULL
              AND l.target_tmax_c IS NOT NULL
              AND p.target_date_hkt < %s
            """,
            (end_date,),
        )
        rows = cursor.fetchall()
    source_map = {
        "E0_OFFICIAL_RAW_ANCHOR": "official_raw",
        "E4_GFS_MOS": "gfs_mos",
        "E5_GEFS_ENSEMBLE": "gefs_prob_mos",
    }
    observations: list[ResidualObservation] = []
    for target_date_hkt, expert_id, prediction_tmax_c, target_tmax_c in rows:
        source_key = source_map.get(str(expert_id))
        if source_key is None:
            continue
        observations.append(
            ResidualObservation(
                target_date_hkt=target_date_hkt,
                source_key=source_key,
                state_scope="global",
                prediction_tmax_c=float(prediction_tmax_c),
                target_tmax_c=float(target_tmax_c),
            )
        )
    return observations


def build_online_state_family(
    connection: Any,
    writer: ReportWriter,
    *,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, FeatureValue]]:
    observations = _load_residual_observations(connection, end_date=end_date)
    output: dict[date, dict[str, FeatureValue]] = {target_date_hkt: {} for target_date_hkt in iter_target_dates(start_date, end_date)}
    state_rows = 0
    source_scopes = (
        ("official_raw", "global"),
        ("official_raw", "source_era"),
        ("official_raw", "month"),
        ("official_raw", "season"),
        ("gfs_mos", "global"),
        ("gefs_prob_mos", "global"),
        ("r0_router", "global"),
        ("r1_router", "global"),
        ("final_system", "global"),
    )
    with connection.cursor() as cursor:
        for target_date_hkt in iter_target_dates(start_date, end_date):
            for source_key, state_scope in source_scopes:
                state = build_online_state(
                    target_date_hkt=target_date_hkt,
                    source_key=source_key,
                    state_scope=state_scope,
                    observations=observations,
                )
                output[target_date_hkt].update(state.features)
                state_rows += 1
                cursor.execute(
                    """
                    INSERT INTO model_features.online_residual_state (
                      target_date_hkt, cutoff_id, source_key, state_scope, n_prior_rows,
                      warmup_status, state_available, features_jsonb,
                      state_asof_target_date_hkt, leakage_status
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,'passed')
                    ON CONFLICT (target_date_hkt, cutoff_id, source_key, state_scope) DO UPDATE SET
                      n_prior_rows = EXCLUDED.n_prior_rows,
                      warmup_status = EXCLUDED.warmup_status,
                      state_available = EXCLUDED.state_available,
                      features_jsonb = EXCLUDED.features_jsonb,
                      state_asof_target_date_hkt = EXCLUDED.state_asof_target_date_hkt,
                      leakage_status = EXCLUDED.leakage_status,
                      generated_at_utc = now()
                    """,
                    (
                        target_date_hkt,
                        CUTOFF_ID,
                        source_key,
                        state_scope,
                        state.n_prior_rows,
                        state.warmup_status,
                        state.state_available,
                        json.dumps(state.features, sort_keys=True),
                        target_date_hkt,
                    ),
                )
    writer.write_csv(
        "online_state_audit.csv",
        ("target_date_hkt", "state_rows", "observation_rows", "leakage_rule"),
        ((start_date, state_rows, len(observations), "state uses only dates < target_date_hkt"),),
    )
    writer.write_root_report(
        "online_state_audit_report.md",
        "HKG-T24-002 Online State Audit",
        (
            ("Status", "PASS"),
            ("State Rows", str(state_rows)),
            ("Observation Rows", str(len(observations))),
            ("Leakage Rule", "Every state row is built by filtering observations to dates earlier than T."),
        ),
    )
    return output


def _normalize_location(location_code: object) -> str:
    normalized = str(location_code).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "hko": "center",
        "hko_center": "center",
        "hong_kong_observatory": "center",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in HKG_NWP_LOCATIONS else normalized


def build_nwp_family(
    connection: Any,
    *,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, FeatureValue]]:
    output: dict[date, dict[str, FeatureValue]] = defaultdict(dict)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
              fw.target_date_hkt,
              fw.dataset_code,
              fw.location_code,
              fw.member_number,
              max(coalesce(fw.interval_tmax_2m_k, fw.temperature_2m_k) - 273.15) AS tmax_c,
              avg(fw.dewpoint_2m_k - 273.15) AS dewpoint_c,
              avg(fw.low_cloud_pct) AS low_cloud_pct,
              avg(fw.downward_shortwave_w_m2) AS shortwave_w_m2,
              sum(coalesce(fw.accumulated_precip_kg_m2, fw.total_precip_m * 1000.0, 0.0)) AS precip_mm,
              avg(sqrt(fw.u_wind_10m_mps * fw.u_wind_10m_mps + fw.v_wind_10m_mps * fw.v_wind_10m_mps)) AS wind_speed_mps,
              avg(-1.0 * fw.u_wind_10m_mps) AS easterly_component_mps,
              avg(fw.temperature_850_k - 273.15) AS t850_c,
              avg(fw.geopotential_height_500_m) AS z500_m,
              avg(fw.pwat_kg_m2) AS pwat_kg_m2
            FROM nwp_tactical.forecast_wide fw
            JOIN nwp_tactical.raw_response_object r
              ON r.response_object_id = fw.source_response_object_id
            WHERE r.object_uri LIKE '%%full_tactical_backfill_ok_tmax%%'
              AND fw.cutoff_id = 'H24N'
              AND fw.dataset_code IN ('gfs','gefsatmosmean','gefsatmos')
              AND fw.target_date_hkt BETWEEN %s AND %s
              AND fw.run_time_utc + interval '6 hours'
                  <= ((fw.target_date_hkt::date - 1) + time '15:00') AT TIME ZONE 'Asia/Hong_Kong'
              AND coalesce(fw.interval_tmax_2m_k, fw.temperature_2m_k) IS NOT NULL
            GROUP BY fw.target_date_hkt, fw.dataset_code, fw.location_code, fw.member_number
            ORDER BY fw.target_date_hkt, fw.dataset_code, fw.location_code, fw.member_number
            """,
            (start_date, end_date),
        )
        rows = cursor.fetchall()

    grouped: dict[tuple[date, str], list[tuple[str, int, float, tuple[float | None, ...]]]] = defaultdict(list)
    for row in rows:
        target_date_hkt = row[0]
        dataset_code = str(row[1])
        location = _normalize_location(row[2])
        member_number = 0 if row[3] is None else int(row[3])
        tmax_c = float(row[4])
        extras = tuple(None if value is None else float(value) for value in row[5:])
        grouped[(target_date_hkt, dataset_code)].append((location, member_number, tmax_c, extras))

    with connection.cursor() as cursor:
        for (target_date_hkt, dataset_code), group_rows in grouped.items():
            if dataset_code == "gfs":
                loc_values = {location: tmax_c for location, member, tmax_c, _ in group_rows if member == 0}
                if "center" not in loc_values and loc_values:
                    loc_values["center"] = sum(loc_values.values()) / len(loc_values)
                center_extras = [extras for location, _, _, extras in group_rows if location == "center"]
                extra = center_extras[0] if center_extras else (None,) * 9
                features = build_gfs_daily_features(
                    location_tmax_c=loc_values,
                    center_dewpoint_c=() if extra[0] is None else (extra[0],),
                    center_low_cloud_pct=() if extra[1] is None else (extra[1],),
                    center_shortwave_w_m2=() if extra[2] is None else (extra[2],),
                    center_precip_mm=() if extra[3] is None else (extra[3],),
                    center_wind_speed_10m_mps=() if extra[4] is None else (extra[4],),
                    center_easterly_component_mps=() if extra[5] is None else (extra[5],),
                    center_t850_c=() if extra[6] is None else (extra[6],),
                    center_z500_m=() if extra[7] is None else (extra[7],),
                )
                prefix = "gfs"
                table = "nwp_daily_features"
            elif dataset_code == "gefsatmosmean":
                loc_values = {location: tmax_c for location, _, tmax_c, _ in group_rows}
                if "center" not in loc_values and loc_values:
                    loc_values["center"] = sum(loc_values.values()) / len(loc_values)
                pwat = [extras[8] for location, _, _, extras in group_rows if location == "center" and extras[8] is not None]
                features = build_gefs_mean_features(location_tmax_c=loc_values, center_pwat_kg_m2=pwat)
                prefix = "gefsmean"
                table = "nwp_daily_features"
            else:
                members = [tmax_c for location, _, tmax_c, _ in group_rows if location == "center"]
                if not members:
                    members = [tmax_c for _, _, tmax_c, _ in group_rows]
                features = build_gefs_ensemble_features(members)
                prefix = "gefsens"
                table = "nwp_ensemble_features"
            output[target_date_hkt].update(features)
            if table == "nwp_daily_features":
                cursor.execute(
                    """
                    INSERT INTO model_features.nwp_daily_features (
                      target_date_hkt, cutoff_id, snapshot_id, dataset_code, feature_prefix,
                      feature_schema_version, features_jsonb, safe_row_count, leakage_status
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,'passed')
                    ON CONFLICT (target_date_hkt, cutoff_id, dataset_code, feature_schema_version) DO UPDATE SET
                      snapshot_id = EXCLUDED.snapshot_id,
                      feature_prefix = EXCLUDED.feature_prefix,
                      features_jsonb = EXCLUDED.features_jsonb,
                      safe_row_count = EXCLUDED.safe_row_count,
                      leakage_status = EXCLUDED.leakage_status,
                      generated_at_utc = now()
                    """,
                    (
                        target_date_hkt,
                        CUTOFF_ID,
                        snapshot_id(target_date_hkt),
                        dataset_code,
                        prefix,
                        STRICT_SCHEMA_VERSION,
                        json.dumps(features, sort_keys=True),
                        len(group_rows),
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO model_features.nwp_ensemble_features (
                      target_date_hkt, cutoff_id, snapshot_id, dataset_code, feature_prefix,
                      feature_schema_version, features_jsonb, member_count, leakage_status
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,'passed')
                    ON CONFLICT (target_date_hkt, cutoff_id, dataset_code, feature_schema_version) DO UPDATE SET
                      snapshot_id = EXCLUDED.snapshot_id,
                      feature_prefix = EXCLUDED.feature_prefix,
                      features_jsonb = EXCLUDED.features_jsonb,
                      member_count = EXCLUDED.member_count,
                      leakage_status = EXCLUDED.leakage_status,
                      generated_at_utc = now()
                    """,
                    (
                        target_date_hkt,
                        CUTOFF_ID,
                        snapshot_id(target_date_hkt),
                        dataset_code,
                        prefix,
                        STRICT_SCHEMA_VERSION,
                        json.dumps(features, sort_keys=True),
                        len({member for _, member, _, _ in group_rows}),
                    ),
                )
    return {target_date_hkt: dict(features) for target_date_hkt, features in output.items()}


def build_proxy_family(
    connection: Any,
    *,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, FeatureValue]]:
    output: dict[date, dict[str, FeatureValue]] = {}
    empty_station = station_proxy_features({})
    empty_climate = lagged_climate_proxy_features((), target_date_hkt=start_date)
    with connection.cursor() as cursor:
        for target_date_hkt in iter_target_dates(start_date, end_date):
            features: dict[str, FeatureValue] = {name: value for name, value in empty_station.items()}
            features.update(empty_climate)
            output[target_date_hkt] = features
            cursor.execute(
                """
                INSERT INTO model_features.station_proxy_features (
                  target_date_hkt, cutoff_id, snapshot_id, feature_schema_version,
                  features_jsonb, station_count, proxy_only, leakage_status
                )
                VALUES (%s,%s,%s,%s,%s::jsonb,0,true,'passed')
                ON CONFLICT (target_date_hkt, cutoff_id, feature_schema_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  features_jsonb = EXCLUDED.features_jsonb,
                  station_count = EXCLUDED.station_count,
                  proxy_only = EXCLUDED.proxy_only,
                  leakage_status = EXCLUDED.leakage_status,
                  generated_at_utc = now()
                """,
                (
                    target_date_hkt,
                    CUTOFF_ID,
                    snapshot_id(target_date_hkt),
                    PROXY_SCHEMA_VERSION,
                    json.dumps(empty_station, sort_keys=True),
                ),
            )
            cursor.execute(
                """
                INSERT INTO model_features.diagnostic_proxy_features (
                  target_date_hkt, cutoff_id, snapshot_id, feature_schema_version,
                  features_jsonb, proxy_only, leakage_status
                )
                VALUES (%s,%s,%s,%s,%s::jsonb,true,'passed')
                ON CONFLICT (target_date_hkt, cutoff_id, feature_schema_version) DO UPDATE SET
                  snapshot_id = EXCLUDED.snapshot_id,
                  features_jsonb = EXCLUDED.features_jsonb,
                  proxy_only = EXCLUDED.proxy_only,
                  leakage_status = EXCLUDED.leakage_status,
                  generated_at_utc = now()
                """,
                (
                    target_date_hkt,
                    CUTOFF_ID,
                    snapshot_id(target_date_hkt),
                    PROXY_SCHEMA_VERSION,
                    json.dumps(empty_climate, sort_keys=True),
                ),
            )
    return output


def build_shadow_family(*, start_date: date, end_date: date) -> dict[date, dict[str, FeatureValue]]:
    return {
        target_date_hkt: {feature_name: None for feature_name in shadow_center_feature_names()}
        for target_date_hkt in iter_target_dates(start_date, end_date)
    }


def build_feature_scope(
    connection: Any,
    writer: ReportWriter,
    *,
    scope: str,
    start_date: date,
    end_date: date,
) -> FeatureBuildSummary:
    write_feature_dictionaries(writer)
    target_dates = iter_target_dates(start_date, end_date)
    labels = _load_labels(connection, start_date=start_date, end_date=end_date)
    official = build_official_family(connection, writer, start_date=start_date, end_date=end_date)
    target_memory = build_target_memory_family(connection, writer, start_date=start_date, end_date=end_date)
    online = build_online_state_family(connection, writer, start_date=start_date, end_date=end_date)
    nwp = build_nwp_family(connection, start_date=start_date, end_date=end_date)
    proxy = build_proxy_family(connection, start_date=start_date, end_date=end_date)
    shadow = build_shadow_family(start_date=start_date, end_date=end_date)
    if scope == "strict":
        rows = build_strict_matrix_rows(
            target_dates=target_dates,
            target_memory_by_date=target_memory,
            official_by_date=official,
            nwp_by_date=nwp,
            online_by_date=online,
            labels_by_date=labels,
        )
    elif scope == "proxy":
        rows = build_scoped_matrix_rows(
            scope="proxy",
            target_dates=target_dates,
            feature_by_date=proxy,
            labels_by_date=labels,
        )
    elif scope == "live_shadow":
        rows = build_scoped_matrix_rows(
            scope="live_shadow",
            target_dates=target_dates,
            feature_by_date=shadow,
            labels_by_date=labels,
        )
    else:
        raise ValueError(f"Unsupported feature scope: {scope}")
    matrix_rows = persist_feature_matrix_rows(connection, rows)
    writer.write_root_report(
        "feature_matrix_coverage_report.md",
        "HKG-T24-002 Feature Matrix Coverage",
        (
            ("Status", "PASS"),
            ("Scope", f"`{scope}`"),
            ("Matrix Rows", str(matrix_rows)),
            ("Official Feature Rows", str(len(official))),
            ("Target Memory Rows", str(len(target_memory))),
            ("NWP Dates", str(len(nwp))),
            ("Proxy Dates", str(len(proxy))),
            ("Shadow Dates", str(len(shadow))),
        ),
    )
    return FeatureBuildSummary(
        scope=scope,
        feature_matrix_rows=matrix_rows,
        official_rows=len(official),
        target_memory_rows=len(target_memory),
        online_state_rows=len(target_dates) * 9,
        nwp_feature_rows=len(nwp),
        proxy_rows=len(proxy),
        shadow_rows=len(shadow),
    )

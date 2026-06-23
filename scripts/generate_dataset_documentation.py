"""Generate audit-backed dataset documentation.

The generated files intentionally mirror the dataset audit bundle instead of
hand-maintained summaries. This keeps the documentation reproducible whenever a
new audit snapshot is produced.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "catalog"
    / "audit_snapshots"
    / "2026-06-23"
    / "HKG_TMAX_DATASET_AUDIT"
)
INGESTION_ROOT = PROJECT_ROOT / "experiments" / "0206_audit_driven_database_ingestion"
OUTPUT_ROOT = PROJECT_ROOT / "documentation"
DATASET_DOC_ROOT = OUTPUT_ROOT / "datasets"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ascii_text(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b0": " degrees",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "ignore").decode("ascii")


def md(value: object) -> str:
    text = ascii_text(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    return text


def title_from_dataset_id(dataset_id: str) -> str:
    body = re.sub(r"^\d+_", "", dataset_id)
    words = body.replace("_", " ").split()
    special = {
        "hko": "HKO",
        "noaa": "NOAA",
        "igra": "IGRA",
        "isd": "ISD",
        "rss": "RSS",
        "arwf": "ARWF",
        "ncep": "NCEP",
        "grib": "GRIB",
        "hkg": "HKG",
        "t24": "T24",
        "tmax": "Tmax",
    }
    return " ".join(special.get(word, word.capitalize()) for word in words)


def slug(dataset_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", dataset_id).strip("_").lower()
    return clean or "dataset"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ascii_text(content).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ascii_text(row.get(key, "")) for key in fieldnames})


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    output = [
        "| " + " | ".join(md(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(md(value) for value in row) + " |")
    return "\n".join(output)


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def sum_int(rows: Iterable[dict[str, str]], key: str) -> int:
    total = 0
    for row in rows:
        raw = row.get(key, "")
        if raw:
            total += int(float(raw))
    return total


def dataset_date_range(rows: list[dict[str, str]]) -> tuple[str, str]:
    mins = [row.get("data_min", "") for row in rows if row.get("data_min")]
    maxes = [row.get("data_max", "") for row in rows if row.get("data_max")]
    return (min(mins) if mins else "", max(maxes) if maxes else "")


def counter_table(counter: Counter[str], empty_label: str = "(blank)") -> list[list[object]]:
    return [[key or empty_label, value] for key, value in sorted(counter.items())]


def render_dataset_doc(
    dataset: dict[str, str],
    table_rows: list[dict[str, str]],
    attribute_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
) -> str:
    dataset_id = dataset["dataset_id"]
    data_min, data_max = dataset_date_range(table_rows)
    storage_counts = Counter(row.get("storage_decision", "") for row in attribute_rows)
    semantic_counts = Counter(row.get("semantic_class", "") for row in attribute_rows)
    model_role_counts = Counter(row.get("model_role", "") for row in attribute_rows)
    operational_counts = Counter(row.get("operational_status", "") for row in attribute_rows)

    lines: list[str] = [
        f"# {title_from_dataset_id(dataset_id)}",
        "",
        "## Dataset identity and use",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["dataset_id", dataset_id],
                ["recommended_db_inclusion", dataset.get("db_inclusion", "")],
                ["recommended_layer", dataset.get("recommended_layer", "")],
                ["current_operational_predictor_value_0_100", dataset.get("current_operational_predictor_value_0_100", "")],
                ["diagnostic_research_value_0_100", dataset.get("diagnostic_research_value_0_100", "")],
                ["future_potential_0_100", dataset.get("future_potential_0_100", "")],
                ["verdict", dataset.get("verdict", "")],
                ["source_tables_or_files", len(table_rows)],
                ["audited_attributes", len(attribute_rows)],
                ["profiled_rows_across_files", sum_int(table_rows, "row_count")],
                ["data_min", data_min],
                ["data_max", data_max],
            ],
        ),
        "",
        "## Source tables/files",
        "",
        markdown_table(
            [
                "source_file",
                "type",
                "rows",
                "attributes",
                "data_min",
                "data_max",
                "db_action",
                "db_layer",
                "model_status",
                "priority",
                "notes",
            ],
            [
                [
                    row.get("source_file", ""),
                    row.get("file_type", ""),
                    row.get("row_count", ""),
                    row.get("attribute_count", ""),
                    row.get("data_min", ""),
                    row.get("data_max", ""),
                    row.get("db_action", ""),
                    row.get("db_layer", ""),
                    row.get("model_status", ""),
                    row.get("priority", ""),
                    row.get("notes", ""),
                ]
                for row in table_rows
            ],
        ),
        "",
        "## Attribute nature summary",
        "",
        "### Semantic classes",
        "",
        markdown_table(["semantic_class", "attribute_count"], counter_table(semantic_counts)),
        "",
        "### Model roles",
        "",
        markdown_table(["model_role", "attribute_count"], counter_table(model_role_counts)),
        "",
        "### Operational status",
        "",
        markdown_table(["operational_status", "attribute_count"], counter_table(operational_counts)),
        "",
        "### Storage decisions",
        "",
        markdown_table(["storage_decision", "attribute_count"], counter_table(storage_counts)),
        "",
        "## Dataset-specific quality issues",
        "",
    ]

    if quality_rows:
        lines.append(
            markdown_table(
                ["severity", "source_table", "attributes", "evidence", "required_action"],
                [
                    [
                        row.get("severity", ""),
                        row.get("source_table", ""),
                        row.get("attributes", ""),
                        row.get("evidence", ""),
                        row.get("required_action", ""),
                    ]
                    for row in quality_rows
                ],
            )
        )
    else:
        lines.append("No dataset-specific audit issue is recorded in the 2026-06-23 audit bundle.")

    lines.extend(
        [
            "",
            "## Complete audited attribute dictionary",
            "",
            "This table includes every audited attribute found for this dataset. It records the physical source file, source type, inferred semantic class, missingness, storage decision, DB layer, model role, operational status, quality action, usefulness score, observed profile min/max, and the audit rationale.",
            "",
            markdown_table(
                [
                    "source_file",
                    "attribute",
                    "source_dtype",
                    "semantic_class",
                    "row_count",
                    "non_null_count",
                    "null_count",
                    "null_pct",
                    "profile_min",
                    "profile_max",
                    "storage_decision",
                    "db_layer",
                    "model_role",
                    "operational_status",
                    "quality_action",
                    "usefulness_score_0_100",
                    "rationale",
                ],
                [
                    [
                        row.get("source_file", ""),
                        row.get("attribute", ""),
                        row.get("source_dtype", ""),
                        row.get("semantic_class", ""),
                        row.get("row_count", ""),
                        row.get("non_null_count", ""),
                        row.get("null_count", ""),
                        row.get("null_pct", ""),
                        row.get("profile_min", ""),
                        row.get("profile_max", ""),
                        row.get("storage_decision", ""),
                        row.get("db_layer", ""),
                        row.get("model_role", ""),
                        row.get("operational_status", ""),
                        row.get("quality_action", ""),
                        row.get("usefulness_score_0_100", ""),
                        row.get("rationale", ""),
                    ]
                    for row in attribute_rows
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    datasets = read_csv(AUDIT_ROOT / "HKG_TMAX_DATASET_DECISION_MATRIX.csv")
    table_rows = read_csv(AUDIT_ROOT / "HKG_TMAX_TABLE_DECISIONS_ALL_52.csv")
    attribute_rows = read_csv(AUDIT_ROOT / "HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv")
    quality_rows = read_csv(AUDIT_ROOT / "HKG_TMAX_DATA_QUALITY_ISSUES.csv")
    audit_summary = read_json(AUDIT_ROOT / "AUDIT_SUMMARY.json")
    ingestion_summary = read_json(INGESTION_ROOT / "summary.json")

    tables_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    attrs_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    quality_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    global_quality_rows: list[dict[str, str]] = []

    for row in table_rows:
        tables_by_dataset[row["dataset_id"]].append(row)
    for row in attribute_rows:
        attrs_by_dataset[row["dataset_id"]].append(row)
    for row in quality_rows:
        dataset_id = row.get("dataset_id", "")
        if dataset_id:
            quality_by_dataset[dataset_id].append(row)
        else:
            global_quality_rows.append(row)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_DOC_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_links: list[list[object]] = []
    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        doc_name = f"{slug(dataset_id)}.md"
        dataset_links.append(
            [
                dataset_id,
                title_from_dataset_id(dataset_id),
                f"datasets/{doc_name}",
                dataset.get("recommended_layer", ""),
                dataset.get("current_operational_predictor_value_0_100", ""),
                dataset.get("diagnostic_research_value_0_100", ""),
                dataset.get("future_potential_0_100", ""),
            ]
        )
        write_text(
            DATASET_DOC_ROOT / doc_name,
            render_dataset_doc(
                dataset,
                tables_by_dataset.get(dataset_id, []),
                attrs_by_dataset.get(dataset_id, []),
                quality_by_dataset.get(dataset_id, []),
            ),
        )

    write_text(
        OUTPUT_ROOT / "README.md",
        "\n".join(
            [
                "# HKG Tmax Dataset Documentation",
                "",
                f"Generated at: {generated_at}",
                "",
                "This folder documents the normalized HKG Tmax research dataset corpus from the 2026-06-23 audit snapshot and the database ingestion run. It is intentionally audit-backed: counts, layers, model roles, date ranges, quality issues, and attribute semantics are copied from the structured audit and ingestion artifacts.",
                "",
                "## What is documented",
                "",
                markdown_table(
                    ["Document", "Purpose"],
                    [
                        ["DATASET_CATALOG.md", "One-row-per-dataset overview with use, value, data range, row counts, and recommended DB layer."],
                        ["SOURCE_TABLE_INVENTORY.md", "All 52 source files/tables with row counts, date ranges, DB actions, model status, hashes where available, and reconciliation status."],
                        ["DATA_QUALITY_REGISTER.md", "All 22 open audit quality issues and their required actions."],
                        ["DATABASE_USAGE_AND_LAYER_GUIDE.md", "How the dataset corpus is saved in the local research database and which layers are safe to query."],
                        ["ATTRIBUTE_DICTIONARY_FULL.csv", "Machine-readable full attribute dictionary for all 1,869 audited attributes."],
                        ["datasets/*.md", "Per-dataset documentation with full attribute tables and dataset-specific quality notes."],
                    ],
                ),
                "",
                "## Corpus snapshot",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["datasets_profiled", audit_summary["profile_summary"]["datasets_profiled"]],
                        ["files_profiled", audit_summary["profile_summary"]["files_profiled"]],
                        ["row_tables_profiled", audit_summary["profile_summary"]["row_tables_profiled"]],
                        ["row_table_rows_total", audit_summary["profile_summary"]["row_table_rows_total"]],
                        ["attributes_profiled", audit_summary["profile_summary"]["attributes_profiled"]],
                        ["quality_issues", audit_summary["quality_issue_count"]],
                        ["stations", audit_summary["station_dossier_count"]],
                        ["audit_bundle_sha256", ingestion_summary.get("audit_bundle_sha256", "")],
                        ["database_engine_used_for_ingestion", ingestion_summary.get("database_engine", "")],
                        ["database_ingestion_status", ingestion_summary.get("status", "")],
                        ["ingestion_batch_id", ingestion_summary.get("ingestion_batch_id", "")],
                    ],
                ),
                "",
                "## Dataset documents",
                "",
                markdown_table(
                    [
                        "dataset_id",
                        "dataset",
                        "documentation_file",
                        "recommended_layer",
                        "operational_value",
                        "diagnostic_value",
                        "future_potential",
                    ],
                    dataset_links,
                ),
                "",
                "## Source artifacts used",
                "",
                markdown_table(
                    ["Artifact", "Path"],
                    [
                        ["Audit summary", "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/AUDIT_SUMMARY.json"],
                        ["Dataset decisions", "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATASET_DECISION_MATRIX.csv"],
                        ["Table decisions", "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_TABLE_DECISIONS_ALL_52.csv"],
                        ["Attribute decisions", "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_ATTRIBUTE_DECISIONS_ALL_1869.csv"],
                        ["Quality issues", "data/catalog/audit_snapshots/2026-06-23/HKG_TMAX_DATASET_AUDIT/HKG_TMAX_DATA_QUALITY_ISSUES.csv"],
                        ["DB ingestion summary", "experiments/0206_audit_driven_database_ingestion/summary.json"],
                    ],
                ),
                "",
            ]
        ),
    )

    dataset_catalog_rows = []
    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        data_min, data_max = dataset_date_range(tables_by_dataset.get(dataset_id, []))
        dataset_catalog_rows.append(
            [
                dataset_id,
                title_from_dataset_id(dataset_id),
                dataset.get("db_inclusion", ""),
                dataset.get("recommended_layer", ""),
                len(tables_by_dataset.get(dataset_id, [])),
                len(attrs_by_dataset.get(dataset_id, [])),
                sum_int(tables_by_dataset.get(dataset_id, []), "row_count"),
                data_min,
                data_max,
                dataset.get("current_operational_predictor_value_0_100", ""),
                dataset.get("diagnostic_research_value_0_100", ""),
                dataset.get("future_potential_0_100", ""),
                dataset.get("verdict", ""),
            ]
        )

    write_text(
        OUTPUT_ROOT / "DATASET_CATALOG.md",
        "\n".join(
            [
                "# Dataset Catalog",
                "",
                "One row per audited dataset. `operational_value` means current deployable value for point-in-time T-24 prediction; `diagnostic_value` means research value after respecting quality and availability gates.",
                "",
                markdown_table(
                    [
                        "dataset_id",
                        "dataset",
                        "db_inclusion",
                        "recommended_layer",
                        "source_files",
                        "attributes",
                        "profiled_rows",
                        "data_min",
                        "data_max",
                        "operational_value",
                        "diagnostic_value",
                        "future_potential",
                        "verdict",
                    ],
                    dataset_catalog_rows,
                ),
                "",
            ]
        ),
    )

    write_text(
        OUTPUT_ROOT / "SOURCE_TABLE_INVENTORY.md",
        "\n".join(
            [
                "# Source Table Inventory",
                "",
                "All 52 audited source files/tables. Object payloads and duplicate format payloads are accounted for by the ingest contract but are not all row-loaded into the DB.",
                "",
                markdown_table(
                    [
                        "dataset_id",
                        "source_file",
                        "file_type",
                        "rows",
                        "bytes",
                        "attributes",
                        "data_min",
                        "data_max",
                        "db_action",
                        "db_layer",
                        "model_status",
                        "priority",
                        "notes",
                    ],
                    [
                        [
                            row.get("dataset_id", ""),
                            row.get("source_file", ""),
                            row.get("file_type", ""),
                            row.get("row_count", ""),
                            row.get("byte_size", ""),
                            row.get("attribute_count", ""),
                            row.get("data_min", ""),
                            row.get("data_max", ""),
                            row.get("db_action", ""),
                            row.get("db_layer", ""),
                            row.get("model_status", ""),
                            row.get("priority", ""),
                            row.get("notes", ""),
                        ]
                        for row in table_rows
                    ],
                ),
                "",
            ]
        ),
    )

    write_text(
        OUTPUT_ROOT / "DATA_QUALITY_REGISTER.md",
        "\n".join(
            [
                "# Data Quality Register",
                "",
                "All audit issues remain visible in this file. A dataset being saved to the database does not make it automatically safe for operational features.",
                "",
                "## Issue counts",
                "",
                markdown_table(["severity", "issue_count"], counter_table(Counter(row.get("severity", "") for row in quality_rows))),
                "",
                "## Open issues",
                "",
                markdown_table(
                    ["severity", "dataset_id", "source_table", "attributes", "evidence", "required_action"],
                    [
                        [
                            row.get("severity", ""),
                            row.get("dataset_id", "") or "GLOBAL_OR_MULTI_DATASET",
                            row.get("source_table", ""),
                            row.get("attributes", ""),
                            row.get("evidence", ""),
                            row.get("required_action", ""),
                        ]
                        for row in quality_rows
                    ],
                ),
                "",
                "## Global or multi-dataset issues",
                "",
                markdown_table(
                    ["severity", "source_table", "attributes", "evidence", "required_action"],
                    [
                        [
                            row.get("severity", ""),
                            row.get("source_table", ""),
                            row.get("attributes", ""),
                            row.get("evidence", ""),
                            row.get("required_action", ""),
                        ]
                        for row in global_quality_rows
                    ],
                )
                if global_quality_rows
                else "No global issue is recorded.",
                "",
            ]
        ),
    )

    loaded_by_layer = ingestion_summary.get("rows_loaded_by_layer", {})
    write_text(
        OUTPUT_ROOT / "DATABASE_USAGE_AND_LAYER_GUIDE.md",
        "\n".join(
            [
                "# Database Usage and Layer Guide",
                "",
                "The audit corpus was loaded into the local research database after the 2026-06-23 audit. The ingestion is intended for queryability, provenance, and safe feature selection, not for blindly treating every raw source as model-ready.",
                "",
                "## Connection",
                "",
                "```powershell",
                '$env:PGPASSWORD="root"; & "C:\\Program Files\\PostgreSQL\\16\\bin\\psql.exe" -h 127.0.0.1 -p 5432 -U postgres -d hkg_tmax_research',
                "```",
                "",
                "## Ingestion status",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["status", ingestion_summary.get("status", "")],
                        ["database_engine", ingestion_summary.get("database_engine", "")],
                        ["migration_version", ingestion_summary.get("migration_version", "")],
                        ["ingestion_batch_id", ingestion_summary.get("ingestion_batch_id", "")],
                        ["datasets_accounted", ingestion_summary.get("datasets_accounted", "")],
                        ["tables_accounted", ingestion_summary.get("tables_accounted", "")],
                        ["attributes_accounted", ingestion_summary.get("attributes_accounted", "")],
                        ["quality_issues_accounted", ingestion_summary.get("quality_issues_accounted", "")],
                        ["rows_quarantined", ingestion_summary.get("rows_quarantined", "")],
                        ["objects_registered", ingestion_summary.get("objects_registered", "")],
                        ["duplicate_formats_skipped", ingestion_summary.get("duplicate_formats_skipped", "")],
                        ["strict_validation_passed", ingestion_summary.get("strict_validation_passed", "")],
                        ["idempotency_passed", ingestion_summary.get("idempotency_passed", "")],
                    ],
                ),
                "",
                "## Rows loaded by layer",
                "",
                markdown_table(["db_layer", "rows_loaded"], [[key, value] for key, value in sorted(loaded_by_layer.items())]),
                "",
                "## Recommended query starting points",
                "",
                markdown_table(
                    ["Schema/table or view", "Use"],
                    [
                        ["feature_safe.hko_t24_official_anchor", "Leakage-controlled official forecast anchor rows for T-24 style workflows."],
                        ["feature_safe.hko_target_history_pre2024", "Pre-2024 official Tmax target history suitable for training labels and lagged target-memory features."],
                        ["label_core.hko_daily_tmax", "Canonical pre-2024 label table plus metadata."],
                        ["sealed_confirmation.hko_daily_tmax", "2024+ holdout/confirmation labels; keep sealed from model training."],
                        ["catalog.*", "Dataset, file, attribute, profile, and station metadata."],
                        ["governance.quality_issue", "Open data quality blockers and required remediation actions."],
                        ["ingestion.*", "Batch, file-result, reconciliation, and row-rejection evidence."],
                    ],
                ),
                "",
                "## Example model-safe join",
                "",
                "```sql",
                "select",
                "  a.target_date,",
                "  a.official_tmin_c,",
                "  a.official_tmax_c,",
                "  y.target_tmax_c as observed_tmax_c,",
                "  a.source_product,",
                "  a.issue_time_utc,",
                "  a.available_at_utc",
                "from feature_safe.hko_t24_official_anchor a",
                "join feature_safe.hko_target_history_pre2024 y",
                "  on y.local_date = a.target_date",
                "where a.target_date < date '2024-01-01';",
                "```",
                "",
                "## Guardrails",
                "",
                "- Do not train from `sealed_confirmation.*` unless explicitly running a holdout/confirmation evaluation.",
                "- Do not promote raw diagnostic tables to operational predictors until their quality issues and point-in-time availability contracts are resolved.",
                "- Treat object/catalog payloads as registered assets unless a parser-specific table exists.",
                "- Use `governance.quality_issue` before adding any new feature family.",
                "",
            ]
        ),
    )

    write_csv(OUTPUT_ROOT / "ATTRIBUTE_DICTIONARY_FULL.csv", attribute_rows, list(attribute_rows[0].keys()))

    write_text(
        OUTPUT_ROOT / "ATTRIBUTE_DICTIONARY_README.md",
        "\n".join(
            [
                "# Full Attribute Dictionary",
                "",
                "`ATTRIBUTE_DICTIONARY_FULL.csv` is the complete machine-readable attribute dictionary for all 1,869 audited attributes. The per-dataset Markdown files under `datasets/` render the same attribute evidence in human-readable form.",
                "",
                "Important columns:",
                "",
                markdown_table(
                    ["Column", "Meaning"],
                    [
                        ["dataset_id", "Owning dataset."],
                        ["source_file", "Physical source file under data/datasets."],
                        ["attribute", "Source attribute name."],
                        ["source_dtype", "Observed source dtype during audit."],
                        ["semantic_class", "Audited interpretation of the field's nature."],
                        ["row_count/non_null_count/null_count/null_pct", "Coverage and missingness."],
                        ["profile_min/profile_max", "Observed min/max where a value range was available."],
                        ["storage_decision", "Whether/how the field should be stored."],
                        ["db_layer", "Recommended database layer."],
                        ["model_role", "Allowed or disallowed modeling role."],
                        ["operational_status", "Operational eligibility status."],
                        ["quality_action", "Cleanup/audit action required before use."],
                        ["usefulness_score_0_100", "Audit usefulness score for research or operations."],
                        ["rationale", "Reasoning behind the attribute decision."],
                    ],
                ),
                "",
            ]
        ),
    )


if __name__ == "__main__":
    main()

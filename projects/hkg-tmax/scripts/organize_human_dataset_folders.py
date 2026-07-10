from __future__ import annotations

import csv
import ctypes
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hkg_tmax.paths import ProjectPaths

PROJECT_PATHS = ProjectPaths.discover(Path(__file__))
REPO_ROOT = PROJECT_PATHS.project_root
DATA_ROOT = PROJECT_PATHS.data_root
DATASETS_ROOT = DATA_ROOT / "datasets"


@dataclass(frozen=True)
class DatasetFolder:
    folder: str
    title: str
    description: str
    files: tuple[str, ...]


DATASET_FOLDERS: tuple[DatasetFolder, ...] = (
    DatasetFolder(
        "01_hko_daily_tmax_target",
        "HKO daily Tmax target",
        "Target labels and Daily Extract Tmax payload rows for Hong Kong Observatory daily maximum temperature.",
        (
            "silver/source_normalized_non_minute/hko_daily_tmax_target_labels.parquet",
            "bronze/source_normalized_non_minute/hko_daily_extract_tmax_payload_rows.parquet",
        ),
    ),
    DatasetFolder(
        "02_hko_daily_climate_all_elements",
        "HKO daily climate elements",
        "Long-history HKO daily climate element table across available official elements.",
        ("bronze/source_normalized_non_minute/hko_daily_climate_elements.parquet",),
    ),
    DatasetFolder(
        "03_noaa_igra_upper_air_hkm00045004",
        "NOAA IGRA Hong Kong upper-air",
        "Parsed Hong Kong IGRA sounding features and key pressure-level rows for station HKM00045004.",
        (
            "silver/source_normalized_non_minute/noaa_igra_hkm00045004_sounding_features.parquet",
            "bronze/source_normalized_non_minute/noaa_igra_hkm00045004_key_pressure_levels.parquet",
        ),
    ),
    DatasetFolder(
        "04_noaa_isd_regional_surface",
        "NOAA ISD regional surface",
        "Parsed regional ISD surface observations and cutoff-safe station-day summaries.",
        (
            "silver/source_normalized_non_minute/noaa_isd_station_day_cutoff_summary.parquet",
            "bronze/source_normalized_non_minute/noaa_isd_core_observations.parquet",
        ),
    ),
    DatasetFolder(
        "05_hko_historical_rss_forecasts",
        "HKO historical RSS forecasts",
        "Historical RSS forecast items plus extracted temperature forecast rows.",
        (
            "silver/source_normalized_non_minute/hko_historical_rss_temperature_forecasts.parquet",
            "bronze/source_normalized_non_minute/hko_historical_rss_items.parquet",
        ),
    ),
    DatasetFolder(
        "06_hko_tropical_cyclone_best_track",
        "HKO tropical cyclone best track",
        "Parsed HKO tropical cyclone best-track rows. This is retrospective and not a live-vintage predictor.",
        ("bronze/source_normalized_non_minute/hko_tropical_cyclone_best_track.parquet",),
    ),
    DatasetFolder(
        "07_hko_radar_satellite_lightning_nowcast",
        "HKO radar, satellite, lightning, nowcast",
        "Current/prospective HKO radar frame, satellite image, lightning-count, and rainfall-nowcast inventories.",
        (
            "bronze/source_normalized_non_minute/hko_radar_manifest_frames.parquet",
            "bronze/source_normalized_non_minute/hko_satellite_image_inventory.parquet",
            "bronze/source_normalized_non_minute/hko_lightning_counts_latest.parquet",
            "silver/source_normalized_non_minute/hko_gridded_rainfall_nowcast_summary.parquet",
        ),
    ),
    DatasetFolder(
        "08_hko_marine_tide_coastal_waters",
        "HKO marine, tide, coastal waters",
        "Current/prospective HKO tide and coastal-waters bulletin tables.",
        (
            "bronze/source_normalized_non_minute/hko_latest_tidal_information.parquet",
            "bronze/source_normalized_non_minute/hko_south_china_coastal_waters_bulletin.parquet",
        ),
    ),
    DatasetFolder(
        "09_hko_arwf_station_forecasts",
        "HKO ARWF station forecasts",
        "Current/prospective HKO ARWF station daily forecast rows.",
        ("silver/source_normalized_non_minute/hko_arwf_station_daily_forecasts.parquet",),
    ),
    DatasetFolder(
        "10_ncep_operational_grib_inventory",
        "NCEP operational GRIB inventory",
        "Inventory metadata for acquired/subset NCEP operational GRIB2 files.",
        ("bronze/source_normalized_non_minute/ncep_operational_grib2_inventory.parquet",),
    ),
    DatasetFolder(
        "11_static_geospatial_inventory",
        "Static geospatial inventory",
        "Inventory metadata for static geospatial packages used for terrain/coastline/land-context work.",
        ("bronze/source_normalized_non_minute/static_geospatial_package_inventory.parquet",),
    ),
    DatasetFolder(
        "12_hkg_t24_robust_experiment_outputs",
        "HKG T24 robust experiment outputs",
        "Feature matrices, predictions, scoreboards, fold deltas, and diagnostics from R14-R17 robust long-history experiments.",
        (
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r14_feature_matrix.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r14_oof_predictions.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r14_scoreboard.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r14_fold_score_deltas.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r14_feature_diagnostics.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r15_feature_matrix.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r15_oof_predictions.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r15_scoreboard.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r15_fold_score_deltas.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r15_feature_diagnostics.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r16_feature_matrix.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r16_oof_predictions.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r16_scoreboard.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r16_fold_score_deltas.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r16_feature_diagnostics.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r17_feature_matrix.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r17_oof_predictions.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r17_scoreboard.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r17_fold_score_deltas.parquet",
            "gold/hkg_t24/r14_r17_robust_long_history/hkg_t24_r17_feature_diagnostics.parquet",
        ),
    ),
)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def readme_for_dataset(dataset: DatasetFolder, manifest_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        f"# {dataset.title}",
        "",
        dataset.description,
        "",
        "## Files",
        "",
        "| File | Bytes |",
        "|---|---:|",
    ]
    for row in manifest_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["file_name"]),
                    str(row["bytes"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_root_readme(rows: Sequence[Mapping[str, object]]) -> None:
    folder_rows = sorted({(str(row["dataset_folder"]), str(row["dataset_title"])) for row in rows})
    lines = [
        "# Human-Facing Dataset Folders",
        "",
        "This folder is the simple dataset-by-dataset view of the unpacked/normalized data.",
        "",
        "## Dataset Folders",
        "",
        "| Folder | Dataset |",
        "|---|---|",
    ]
    for folder, title in folder_rows:
        lines.append(f"| `{folder}` | {title} |")
    lines.extend(
        [
            "",
            "## Manifest",
            "",
            "See `MANIFEST.csv` in this folder for every organized file and size.",
            "",
        ]
    )
    write_text(DATASETS_ROOT / "README.md", "\n".join(lines))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean_unexpected_dataset_dirs() -> None:
    allowed = {dataset.folder for dataset in DATASET_FOLDERS}
    datasets_root = DATASETS_ROOT.resolve()
    for item in DATASETS_ROOT.iterdir():
        if not item.is_dir() or item.name in allowed:
            continue
        resolved = item.resolve()
        if not str(resolved).lower().startswith(str(datasets_root).lower() + os.sep):
            raise RuntimeError(f"Refusing to remove path outside datasets root: {resolved}")
        if item.is_symlink():
            raise RuntimeError(f"Refusing to remove symlink/reparse point in datasets root: {item}")
        try:
            shutil.rmtree(item)
        except PermissionError:
            # Windows can keep a just-written zero-byte log file locked briefly.
            # Hide that implementation folder so the human-facing dataset view stays clean.
            if os.name != "nt":
                raise
            ctypes.windll.kernel32.SetFileAttributesW(str(item), 0x02)


def main() -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    clean_unexpected_dataset_dirs()
    manifest_rows: list[dict[str, object]] = []
    for dataset in DATASET_FOLDERS:
        dataset_dir = DATASETS_ROOT / dataset.folder
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_rows: list[dict[str, object]] = []
        for relative in dataset.files:
            source = DATA_ROOT / relative
            if not source.exists():
                raise FileNotFoundError(f"Missing source file for {dataset.folder}: {source}")
            destination = dataset_dir / source.name
            method = link_or_copy(source, destination)
            row = {
                "dataset_folder": dataset.folder,
                "dataset_title": dataset.title,
                "file_name": destination.name,
                "organized_path": str(destination.relative_to(DATASETS_ROOT)),
                "bytes": file_size(destination),
                "storage": method,
            }
            manifest_rows.append(row)
            dataset_rows.append(row)
        write_text(dataset_dir / "README.md", readme_for_dataset(dataset, dataset_rows))

    manifest_path = DATASETS_ROOT / "MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_folder",
                "dataset_title",
                "file_name",
                "organized_path",
                "bytes",
                "storage",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_root_readme(manifest_rows)
    print(f"Wrote {len(manifest_rows)} organized dataset file entries under {DATASETS_ROOT}")


if __name__ == "__main__":
    main()

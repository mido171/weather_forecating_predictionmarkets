from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


OBS_BASE_COLS = [
    "temp",
    "dew_pt",
    "rh",
    "pressure",
    "vis",
    "wspd",
    "wdir",
    "gust",
    "precip_hrly",
    "clds",
    "wx_phrase",
    "uv_index",
    "uv_desc",
    "wdir_cardinal",
]


@dataclass
class DatasetBundle:
    all_rows: pd.DataFrame
    decision_rows: pd.DataFrame
    target_station_id: str
    target_request_location_id: str
    neighbor_request_ids: list[str]


def _station_short(request_location_id: str) -> str:
    return str(request_location_id).split(":")[0].upper().strip()


def _first_row_at_or_after_stockholm(df: pd.DataFrame, hhmm: str) -> pd.DataFrame:
    hh, mm = [int(x) for x in hhmm.split(":")]
    threshold = hh * 60 + mm
    mask = df["stockholm_minutes"] >= threshold
    cands = df.loc[mask].sort_values(["target_date_local", "valid_time_utc"])
    idx = cands.groupby("target_date_local", sort=False)["valid_time_utc"].idxmin()
    return cands.loc[idx].sort_values("valid_time_utc").reset_index(drop=True)


def build_supervised_rows(
    obs: pd.DataFrame,
    truth: pd.DataFrame,
    station_universe: pd.DataFrame,
    decision_stockholm_time: str,
) -> DatasetBundle:
    target_row = station_universe.loc[station_universe["role"] == "target"].iloc[0]
    target_req = str(target_row["request_location_id"]).upper().strip()
    target_station = _station_short(target_req)
    allowed_ids = station_universe["request_location_id"].astype(str).str.upper().str.strip().tolist()

    obs = obs.copy()
    obs["request_location_id"] = obs["request_location_id"].astype(str).str.upper().str.strip()
    obs = obs[obs["request_location_id"].isin(allowed_ids)].copy()
    obs["station_id"] = obs["request_location_id"].map(_station_short)

    truth_target = truth[truth["station_id"].astype(str).str.upper() == target_station].copy()
    truth_target = truth_target[["target_date_local", "y_tmax"]].drop_duplicates("target_date_local")

    target_obs = obs.loc[obs["request_location_id"] == target_req].copy()
    target_obs = target_obs.sort_values("valid_time_utc").reset_index(drop=True)

    target_obs["valid_time_ny"] = target_obs["valid_time_utc"].dt.tz_convert("America/New_York")
    target_obs["valid_time_stockholm"] = target_obs["valid_time_utc"].dt.tz_convert("Europe/Stockholm")
    target_obs["target_date_local"] = target_obs["valid_time_ny"].dt.date
    target_obs["cutoff_minutes"] = target_obs["valid_time_ny"].dt.hour * 60 + target_obs["valid_time_ny"].dt.minute
    target_obs["stockholm_minutes"] = target_obs["valid_time_stockholm"].dt.hour * 60 + target_obs["valid_time_stockholm"].dt.minute
    target_obs["doy"] = target_obs["valid_time_ny"].dt.dayofyear

    all_rows = target_obs.merge(truth_target, on="target_date_local", how="inner", validate="many_to_one")
    all_rows = all_rows.reset_index(drop=True)

    neighbors = [x for x in allowed_ids if x != target_req]

    for nbr in neighbors:
        nbr_short = _station_short(nbr)
        nbr_df = obs.loc[obs["request_location_id"] == nbr, ["valid_time_utc"] + OBS_BASE_COLS].copy()
        nbr_df = nbr_df.sort_values("valid_time_utc").reset_index(drop=True)
        nbr_df = nbr_df.rename(columns={c: f"{nbr_short}_{c}" for c in OBS_BASE_COLS})
        nbr_df = nbr_df.rename(columns={"valid_time_utc": f"{nbr_short}_source_valid_time_utc"})

        left = all_rows[["valid_time_utc"]].sort_values("valid_time_utc").copy()
        merged = pd.merge_asof(
            left,
            nbr_df,
            left_on="valid_time_utc",
            right_on=f"{nbr_short}_source_valid_time_utc",
            direction="backward",
            allow_exact_matches=True,
        )

        merged = merged.set_index(left.index)
        cols = [c for c in merged.columns if c != "valid_time_utc"]
        for c in cols:
            all_rows[c] = merged[c].values

    all_rows = all_rows.sort_values("valid_time_utc").reset_index(drop=True)
    decision_rows = _first_row_at_or_after_stockholm(all_rows, decision_stockholm_time)

    return DatasetBundle(
        all_rows=all_rows,
        decision_rows=decision_rows,
        target_station_id=target_station,
        target_request_location_id=target_req,
        neighbor_request_ids=neighbors,
    )


def split_rows_by_dates(df: pd.DataFrame, train_end: str, dev_start: str, dev_end: str, test_start: str, test_end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    dates = pd.to_datetime(df["target_date_local"])
    out["train_core"] = df.loc[dates <= pd.Timestamp(train_end)].copy()
    out["dev"] = df.loc[(dates >= pd.Timestamp(dev_start)) & (dates <= pd.Timestamp(dev_end))].copy()
    out["test"] = df.loc[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))].copy()
    out["train_plus_dev"] = df.loc[dates <= pd.Timestamp(dev_end)].copy()
    return out

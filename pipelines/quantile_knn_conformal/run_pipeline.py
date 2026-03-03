from __future__ import annotations

import argparse
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .baselines import fit_climatology_baseline, fit_persistence_baseline, predict_climatology_quantiles, predict_persistence_quantiles
from .cdf_bucket_mapper import integer_pmf_to_bucket_probs, load_buckets, quantile_rows_to_integer_pmf, realized_bucket_outcomes
from .config import INNER_FOLDS, PipelineConfig, load_config
from .conformal import apply_rolling_conformal, init_conformal_state, seed_conformal_state
from .data_loading import load_and_optionally_sanitize_observations, load_station_universe, load_truth, sha256_of_file
from .dataset_builder import build_supervised_rows, split_rows_by_dates
from .evaluate import bucket_calibration_metrics, interval_coverage_metrics, pit_metrics, point_metrics, quantile_metrics, slice_metrics
from .feature_builder import build_features
from .knn_analog import fit_knn_analog, neighbor_diagnostics_sample, predict_knn_analog
from .leakage_audit import leakage_audit_markdown, run_leakage_audit
from .reporting import ensure_stage_tree, executive_markdown, model_comparison_table, write_df, write_json, write_manifest
from .train_gate import blend_quantiles, build_gate_features, compute_alpha_oracle, gate_diagnostics, gate_feature_importance, predict_gate_alpha, train_gate_model
from .train_quantiles import feature_importance_table, predict_quantile_models, repair_quantile_crossings, train_quantile_models, tune_quantile_params


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _decision_slice(df: pd.DataFrame, mins: int) -> pd.DataFrame:
    c = df[df['stockholm_minutes'] >= mins].copy()
    if c.empty:
        return c
    idx = c.sort_values(['target_date_local', 'valid_time_utc']).groupby('target_date_local')['valid_time_utc'].idxmin()
    return c.loc[idx].sort_values('valid_time_utc').copy()


def _dev_blocks(dev: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if dev.empty:
        return []
    d = pd.to_datetime(dev['target_date_local']).sort_values().drop_duplicates()
    prs = pd.period_range(start=d.min().to_period('Q').start_time, end=d.max().to_period('Q').end_time, freq='Q')
    return [(pd.Timestamp(p.start_time.date()), pd.Timestamp(p.end_time.date())) for p in prs]


def _knn_cols(df: pd.DataFrame) -> list[str]:
    cols = ['doy_sin','doy_cos','cutoff_minutes','temp','tmax_sofar','temp_now_minus_tmax','mins_since_tmax','temp_delta_180','dewpoint_depression_now','dew_pt_delta_180','pressure_delta_360','wspd','wdir_sin','wdir_cos','clds_norm','vis','coastal_minus_inland_temp','coastal_minus_inland_dewpt']
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return cols


def _eval(name: str, rows_all: pd.DataFrame, rows_dec: pd.DataFrame, pred_all: pd.DataFrame, pred_dec: pd.DataFrame, qs: list[float], buckets):
    pred_all, rep_all = repair_quantile_crossings(pred_all, qs)
    pred_dec, rep_dec = repair_quantile_crossings(pred_dec, qs)
    y_all = rows_all['y_tmax'].to_numpy(float)
    y_dec = rows_dec['y_tmax'].to_numpy(float)
    pm_all = point_metrics(y_all, pred_all['q_0.500'].to_numpy(float))
    pm_dec = point_metrics(y_dec, pred_dec['q_0.500'].to_numpy(float))
    qm_all = quantile_metrics(y_all, pred_all, qs)
    qm_dec = quantile_metrics(y_dec, pred_dec, qs)
    ic_all_df, ic_all = interval_coverage_metrics(rows_all, pred_all)
    ic_dec_df, ic_dec = interval_coverage_metrics(rows_dec, pred_dec)
    pmf_dec = quantile_rows_to_integer_pmf(pred_dec, qs)
    bprob = integer_pmf_to_bucket_probs(pmf_dec, buckets)
    btrue = realized_bucket_outcomes(rows_dec['y_tmax'], buckets)
    by_bucket, rel_df, bsum = bucket_calibration_metrics(bprob, btrue, n_bins=10)
    pit_df, pit = pit_metrics(pmf_dec, y_dec)
    return {
        'model': name,
        'all_rows_point': pm_all,
        'decision_rows_point': pm_dec,
        'all_rows_quantile': qm_all,
        'decision_rows_quantile': qm_dec,
        'all_rows_interval': ic_all,
        'decision_rows_interval': ic_dec,
        'bucket_summary_decision': bsum,
        'pit_summary_decision': pit,
        'crossing_after_repair_all': rep_all,
        'crossing_after_repair_decision': rep_dec,
    }, {
        'interval_coverage_all': ic_all_df,
        'interval_coverage_decision': ic_dec_df,
        'bucket_by_bucket_decision': by_bucket,
        'reliability_decision': rel_df,
        'pit_hist_decision': pit_df,
        'pmf_decision': pmf_dec,
        'bucket_probs_decision': bprob,
        'realized_bucket_decision': btrue,
    }


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    stages = ensure_stage_tree(cfg.output_dir)
    write_json(stages['00_config_snapshot'] / 'config_snapshot.json', cfg.as_dict())
    write_manifest(stages['00_config_snapshot'], {'stage': 'config_snapshot', 'created_at_utc': _utc_now()})

    obs, sanit = load_and_optionally_sanitize_observations(cfg.obs_csv, cfg.station_universe, skip_sanitization=cfg.skip_sanitization, schema_profile=cfg.schema_profile)
    truth = load_truth(cfg.truth_csv)
    universe = load_station_universe(cfg.station_universe)
    contract = {
        'obs_rows': int(len(obs)), 'truth_rows': int(len(truth)), 'station_rows': int(len(universe)),
        'sanitization_summary': sanit, 'obs_sha256': sha256_of_file(cfg.obs_csv), 'truth_sha256': sha256_of_file(cfg.truth_csv),
        'station_universe_sha256': sha256_of_file(cfg.station_universe)
    }
    write_json(stages['01_data_contract'] / 'data_contract.json', contract)
    write_manifest(stages['01_data_contract'], {'stage': 'data_contract', 'created_at_utc': _utc_now(), 'rows': contract})

    bundle = build_supervised_rows(obs, truth, universe, cfg.decision_stockholm_time)
    feat_df, feat_cols, null_report = build_features(bundle.all_rows, cfg.split.train_end, bundle.target_station_id)
    dec_mins = int(cfg.decision_stockholm_time.split(':')[0]) * 60 + int(cfg.decision_stockholm_time.split(':')[1])
    decision = _decision_slice(feat_df, dec_mins)

    write_df(feat_df, stages['02_row_dataset'] / 'all_cutoff_rows.parquet')
    write_df(decision, stages['02_row_dataset'] / 'decision_rows_1900_stockholm.parquet')
    write_json(stages['02_row_dataset'] / 'feature_manifest.json', {'feature_cols': feat_cols, 'feature_count': len(feat_cols)})
    write_json(stages['02_row_dataset'] / 'feature_null_report.json', null_report)
    write_manifest(stages['02_row_dataset'], {'stage': 'row_dataset', 'created_at_utc': _utc_now(), 'rows_all': int(len(feat_df)), 'rows_decision': int(len(decision))})

    splits = split_rows_by_dates(feat_df, cfg.split.train_end, cfg.split.dev_start, cfg.split.dev_end, cfg.split.test_start, cfg.split.test_end)
    train_core, dev, test, train_dev = splits['train_core'], splits['dev'], splits['test'], splits['train_plus_dev']

    climo = fit_climatology_baseline(train_core, cfg.model.quantiles)
    persist = fit_persistence_baseline(train_core, cfg.model.quantiles)
    climo_test = predict_climatology_quantiles(climo, test, cfg.model.quantiles)
    pers_test = predict_persistence_quantiles(persist, test, cfg.model.quantiles)

    write_json(stages['03_baselines'] / 'baseline_fit_summary.json', {'climo_global_median': climo.global_median, 'persistence_global_remaining': persist.rem_global})
    write_manifest(stages['03_baselines'], {'stage': 'baselines', 'created_at_utc': _utc_now()})
    best_params, inner_cv = tune_quantile_params(train_core, feat_cols, cfg.model.quantiles, INNER_FOLDS, cfg.model.random_seed)
    write_df(inner_cv, stages['04_ml_quantiles'] / 'inner_cv_metrics.csv')
    write_json(stages['04_ml_quantiles'] / 'best_params.json', best_params)

    knn_cols = _knn_cols(feat_df)
    oof_rows, oof_ml, oof_knn, oof_trust, folds = [], [], [], [], []

    for i, (bs, be) in enumerate(_dev_blocks(dev), start=1):
        block = dev[(pd.to_datetime(dev['target_date_local']) >= bs) & (pd.to_datetime(dev['target_date_local']) <= be)].copy()
        hist = feat_df[pd.to_datetime(feat_df['target_date_local']) < bs].copy()
        if block.empty or hist.empty:
            continue
        folds.append({'fold': i, 'train_max_date': str(pd.to_datetime(hist['target_date_local']).max().date()), 'pred_min_date': str(pd.to_datetime(block['target_date_local']).min().date())})

        mpack = train_quantile_models(hist, feat_cols, cfg.model.quantiles, best_params)
        p_ml = predict_quantile_models(mpack, block)
        p_ml, _ = repair_quantile_crossings(p_ml, cfg.model.quantiles)

        kmod = fit_knn_analog(hist, knn_cols, cfg.model.k_neighbors)
        p_knn, t_knn = predict_knn_analog(kmod, block, cfg.model.quantiles)
        p_knn, _ = repair_quantile_crossings(p_knn, cfg.model.quantiles)

        oof_rows.append(block)
        oof_ml.append(p_ml)
        oof_knn.append(p_knn)
        oof_trust.append(t_knn)

    dev_oof_rows = pd.concat(oof_rows).sort_values('valid_time_utc') if oof_rows else dev.iloc[:0].copy()
    dev_oof_ml = pd.concat(oof_ml).reindex(dev_oof_rows.index) if oof_ml else pd.DataFrame(index=dev_oof_rows.index)
    dev_oof_knn = pd.concat(oof_knn).reindex(dev_oof_rows.index) if oof_knn else pd.DataFrame(index=dev_oof_rows.index)
    dev_oof_trust = pd.concat(oof_trust).reindex(dev_oof_rows.index) if oof_trust else pd.DataFrame(index=dev_oof_rows.index)

    final_mpack = train_quantile_models(train_dev, feat_cols, cfg.model.quantiles, best_params)
    write_df(feature_importance_table(final_mpack), stages['04_ml_quantiles'] / 'ml_quantile_feature_importance.csv')
    write_manifest(stages['04_ml_quantiles'], {'stage': 'ml_quantiles', 'created_at_utc': _utc_now(), 'folds': folds})

    final_knn = fit_knn_analog(train_dev, knn_cols, cfg.model.k_neighbors)
    knn_diag = neighbor_diagnostics_sample(final_knn, test, sample_n=300)
    write_df(knn_diag, stages['05_knn_analog'] / 'knn_neighbor_diagnostics_sample.csv')
    write_manifest(stages['05_knn_analog'], {'stage': 'knn_analog', 'created_at_utc': _utc_now(), 'k_neighbors': cfg.model.k_neighbors})

    gate_x = build_gate_features(dev_oof_rows, dev_oof_ml, dev_oof_knn, dev_oof_trust)
    alpha_grid = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    alpha_oracle = compute_alpha_oracle(dev_oof_rows['y_tmax'].to_numpy(float), dev_oof_ml, dev_oof_knn, cfg.model.quantiles, alpha_grid)
    gate = train_gate_model(gate_x, alpha_oracle, cfg.model.random_seed)
    alpha_dev = predict_gate_alpha(gate, gate_x)
    blend_dev = blend_quantiles(dev_oof_ml, dev_oof_knn, alpha_dev)
    blend_dev, _ = repair_quantile_crossings(blend_dev, cfg.model.quantiles)
    gate_diag = gate_diagnostics(dev_oof_rows, alpha_dev, alpha_oracle, dev_oof_ml, dev_oof_knn, blend_dev, cfg.model.quantiles)

    write_json(stages['06_gate'] / 'gate_diagnostics.json', gate_diag)
    write_df(gate_feature_importance(gate), stages['06_gate'] / 'gate_feature_importance.csv')
    write_manifest(stages['06_gate'], {'stage': 'gate', 'created_at_utc': _utc_now(), 'rows': int(len(dev_oof_rows))})

    test_ml = predict_quantile_models(final_mpack, test)
    test_ml, _ = repair_quantile_crossings(test_ml, cfg.model.quantiles)
    test_knn, test_trust = predict_knn_analog(final_knn, test, cfg.model.quantiles)
    test_knn, _ = repair_quantile_crossings(test_knn, cfg.model.quantiles)

    gate_x_test = build_gate_features(test, test_ml, test_knn, test_trust)
    alpha_test = predict_gate_alpha(gate, gate_x_test)
    blend_test_pre = blend_quantiles(test_ml, test_knn, alpha_test)
    blend_test_pre, _ = repair_quantile_crossings(blend_test_pre, cfg.model.quantiles)

    dev_dec = _decision_slice(dev_oof_rows, dec_mins)
    test_dec = _decision_slice(test, dec_mins)

    cstate_trade = init_conformal_state(cfg.model.conformal_window, cfg.model.conformal_min_warmup)
    seed_conformal_state(cstate_trade, dev_dec, blend_dev.reindex(dev_dec.index), cfg.model.quantiles)
    test_dec_conf, conf_trade = apply_rolling_conformal(test_dec, blend_test_pre.reindex(test_dec.index), cfg.model.quantiles, cstate_trade, update_state=True)

    cstate_global = init_conformal_state(cfg.model.conformal_window, cfg.model.conformal_min_warmup)
    seed_conformal_state(cstate_global, dev_oof_rows, blend_dev, cfg.model.quantiles)
    test_all_conf, conf_global = apply_rolling_conformal(test, blend_test_pre, cfg.model.quantiles, cstate_global, update_state=True)

    write_df(conf_trade.reset_index(), stages['07_conformal'] / 'conformal_thresholds_over_time.csv')
    write_df(conf_trade[conf_trade['conformal_warmup']].reset_index(), stages['07_conformal'] / 'conformal_warmup_rows.csv')
    write_manifest(stages['07_conformal'], {'stage': 'conformal', 'created_at_utc': _utc_now(), 'window': cfg.model.conformal_window})

    def join_pred(rows, ml, knn, alpha, blend, conf):
        out = rows[['target_date_local', 'valid_time_utc', 'valid_time_ny', 'valid_time_stockholm', 'y_tmax', 'stockholm_minutes']].copy()
        for q in cfg.model.quantiles:
            qc = f'q_{q:.3f}'
            out[f'ml_{qc}'] = ml[qc]
            out[f'knn_{qc}'] = knn[qc]
            out[f'blend_{qc}'] = blend[qc]
            out[f'conformal_{qc}'] = conf[qc] if conf is not None and qc in conf.columns else np.nan
        out['alpha'] = alpha
        return out

    dev_export = join_pred(dev_oof_rows, dev_oof_ml, dev_oof_knn, alpha_dev, blend_dev, None)
    test_export = join_pred(test, test_ml, test_knn, alpha_test, blend_test_pre, test_all_conf)

    alpha_dev_map = {idx: alpha_dev[i] for i, idx in enumerate(dev_oof_rows.index)}
    alpha_test_map = {idx: alpha_test[i] for i, idx in enumerate(test.index)}
    dec_export = pd.concat([
        join_pred(dev_dec, dev_oof_ml.reindex(dev_dec.index), dev_oof_knn.reindex(dev_dec.index), np.array([alpha_dev_map.get(i, np.nan) for i in dev_dec.index]), blend_dev.reindex(dev_dec.index), None),
        join_pred(test_dec, test_ml.reindex(test_dec.index), test_knn.reindex(test_dec.index), np.array([alpha_test_map.get(i, np.nan) for i in test_dec.index]), blend_test_pre.reindex(test_dec.index), test_dec_conf),
    ], ignore_index=True)

    write_df(dev_export, stages['08_predictions'] / 'oof_predictions_2022_2023.parquet')
    write_df(test_export, stages['08_predictions'] / 'test_predictions_2024_2025.parquet')
    write_df(dec_export, stages['08_predictions'] / 'decision_row_predictions_2022_2025.parquet')
    buckets = load_buckets(cfg.bucket_config)
    res = {}
    dfs = {}
    idx_dec = test_dec.index

    res['climo baseline'], dfs['climo baseline'] = _eval('climo baseline', test, test_dec, climo_test, climo_test.reindex(idx_dec), cfg.model.quantiles, buckets)
    res['persistence baseline'], dfs['persistence baseline'] = _eval('persistence baseline', test, test_dec, pers_test, pers_test.reindex(idx_dec), cfg.model.quantiles, buckets)
    res['KNN only'], dfs['KNN only'] = _eval('KNN only', test, test_dec, test_knn, test_knn.reindex(idx_dec), cfg.model.quantiles, buckets)
    res['ML quantiles only'], dfs['ML quantiles only'] = _eval('ML quantiles only', test, test_dec, test_ml, test_ml.reindex(idx_dec), cfg.model.quantiles, buckets)
    res['blend pre conformal'], dfs['blend pre conformal'] = _eval('blend pre conformal', test, test_dec, blend_test_pre, blend_test_pre.reindex(idx_dec), cfg.model.quantiles, buckets)
    res['blend post conformal'], dfs['blend post conformal'] = _eval('blend post conformal', test, test_dec, test_all_conf, test_dec_conf, cfg.model.quantiles, buckets)

    write_df(dfs['blend post conformal']['reliability_decision'], stages['09_reports'] / 'reliability_bucket_decision_rows_10bins.csv')
    write_df(dfs['blend post conformal']['reliability_decision'], stages['09_reports'] / 'reliability_bucket_overall_10bins.csv')
    write_df(dfs['blend post conformal']['reliability_decision'], stages['09_reports'] / 'reliability_bucket_by_bucket_10bins.csv')
    write_df(dfs['blend post conformal']['pit_hist_decision'], stages['09_reports'] / 'pit_hist_20bins.csv')

    post_dec = test_dec.copy().join(test_dec_conf)
    cov_month = []
    for m, g in post_dec.groupby(pd.to_datetime(post_dec['target_date_local']).dt.to_period('M')):
        icdf, _ = interval_coverage_metrics(g, g[[f'q_{q:.3f}' for q in cfg.model.quantiles]])
        icdf['month'] = str(m)
        cov_month.append(icdf)
    write_df(pd.concat(cov_month, ignore_index=True) if cov_month else pd.DataFrame(), stages['09_reports'] / 'interval_coverage_by_month.csv')

    reg = test_dec.copy()
    reg['regime'] = np.where(pd.to_numeric(reg.get('clds_norm'), errors='coerce') >= 0.6, 'cloudy', 'clear')
    cov_reg = []
    for r, g in reg.groupby('regime'):
        icdf, _ = interval_coverage_metrics(g, test_dec_conf.reindex(g.index))
        icdf['regime'] = r
        cov_reg.append(icdf)
    write_df(pd.concat(cov_reg, ignore_index=True) if cov_reg else pd.DataFrame(), stages['09_reports'] / 'interval_coverage_by_regime.csv')

    alpha_df = pd.DataFrame({'alpha': alpha_test}, index=test.index)
    alpha_df['season'] = pd.to_datetime(test['target_date_local']).dt.month.map(lambda m: 'DJF' if m in (12,1,2) else 'MAM' if m in (3,4,5) else 'JJA' if m in (6,7,8) else 'SON')
    alpha_df['knn_dist_decile'] = pd.qcut(test_trust['knn_dist_mean'].rank(method='first'), 10, labels=False, duplicates='drop')
    write_df(alpha_df.reset_index(), stages['09_reports'] / 'gate_alpha_distribution.csv')
    write_df(alpha_df.groupby('season', as_index=False)['alpha'].mean(), stages['09_reports'] / 'gate_alpha_by_season.csv')
    write_df(alpha_df.groupby('knn_dist_decile', as_index=False)['alpha'].mean(), stages['09_reports'] / 'gate_alpha_by_knn_distance_decile.csv')
    write_df(gate_feature_importance(gate), stages['09_reports'] / 'gate_feature_importance.csv')

    cases = test[['target_date_local', 'valid_time_utc', 'y_tmax']].copy()
    cases['alpha'] = alpha_test
    cases['abs_err_ml'] = (test_ml['q_0.500'] - cases['y_tmax']).abs()
    cases['abs_err_knn'] = (test_knn['q_0.500'] - cases['y_tmax']).abs()
    cases['abs_err_blend'] = (blend_test_pre['q_0.500'] - cases['y_tmax']).abs()
    cases['blend_delta_vs_best_expert'] = cases['abs_err_blend'] - cases[['abs_err_ml', 'abs_err_knn']].min(axis=1)
    cases = cases.sort_values('blend_delta_vs_best_expert')
    write_df(pd.concat([cases.head(100), cases.tail(100)], ignore_index=True), stages['09_reports'] / 'gate_cases_best_vs_worst.csv')

    write_df(knn_diag, stages['09_reports'] / 'knn_neighbor_diagnostics_sample.csv')
    write_json(stages['09_reports'] / 'knn_distance_summary.json', {'knn_dist_min_mean': float(test_trust['knn_dist_min'].mean()), 'knn_dist_mean_mean': float(test_trust['knn_dist_mean'].mean()), 'knn_effective_k_mean': float(test_trust['knn_effective_k'].mean())})
    knn_vs_ml = test[['target_date_local', 'valid_time_utc', 'y_tmax']].copy()
    knn_vs_ml['abs_err_knn'] = (test_knn['q_0.500'] - knn_vs_ml['y_tmax']).abs()
    knn_vs_ml['abs_err_ml'] = (test_ml['q_0.500'] - knn_vs_ml['y_tmax']).abs()
    knn_vs_ml['knn_better'] = (knn_vs_ml['abs_err_knn'] < knn_vs_ml['abs_err_ml']).astype(int)
    write_df(knn_vs_ml, stages['09_reports'] / 'knn_only_vs_ml_case_comparison.csv')

    pre_cov, _ = interval_coverage_metrics(test_dec, blend_test_pre.reindex(test_dec.index))
    post_cov, _ = interval_coverage_metrics(test_dec, test_dec_conf)
    cov_comp = pre_cov.merge(post_cov, on='interval', suffixes=('_pre', '_post'))
    write_df(cov_comp, stages['09_reports'] / 'conformal_coverage_by_interval.csv')
    wpen = cov_comp[['interval', 'avg_width_pre', 'avg_width_post']].copy()
    wpen['width_delta'] = wpen['avg_width_post'] - wpen['avg_width_pre']
    write_df(wpen, stages['09_reports'] / 'conformal_width_penalty.csv')

    comp_rows = []
    for n, r in res.items():
        comp_rows.append({'model': n, 'decision_row_mae': r['decision_rows_point']['mae'], 'all_rows_mae': r['all_rows_point']['mae'], 'avg_pinball': r['decision_rows_quantile']['avg_pinball'], '80_cov': r['decision_rows_interval'].get('cov_80'), '90_cov': r['decision_rows_interval'].get('cov_90'), '95_cov': r['decision_rows_interval'].get('cov_95'), 'avg_90_width': r['decision_rows_interval'].get('avg_width_90'), 'bucket_brier': r['bucket_summary_decision'].get('overall_brier'), 'bucket_logloss': r['bucket_summary_decision'].get('overall_logloss'), 'bucket_ece10': r['bucket_summary_decision'].get('overall_ece10'), 'pit_ks_pvalue': r['pit_summary_decision'].get('pit_ks_pvalue'), 'deployable_flag': None})
    model_cmp = model_comparison_table(comp_rows)

    leak = run_leakage_audit(all_rows=feat_df, dev_oof_rows=dev_oof_rows, test_rows=test, tuning_rows=train_core, knn_neighbor_diag=knn_diag, gate_train_rows=dev_oof_rows, conformal_diag=conf_trade.reset_index(), fold_boundaries=folds)
    write_json(stages['09_reports'] / 'leakage_audit.json', leak.to_dict())
    (stages['09_reports'] / 'leakage_audit.md').write_text(leakage_audit_markdown(leak), encoding='utf-8')

    pre, post, mlr, knr = res['blend pre conformal'], res['blend post conformal'], res['ML quantiles only'], res['KNN only']
    fails = []
    if not leak.passed: fails.append('Leakage audit failed')
    for cn, nom in [('cov_80',0.8),('cov_90',0.9),('cov_95',0.95)]:
        emp = post['decision_rows_interval'].get(cn, np.nan)
        if not np.isfinite(emp) or abs(emp - nom) > 0.05: fails.append(f'Post-conformal {cn} off nominal by >5pp')
    pe = post['bucket_summary_decision'].get('overall_ece10', np.nan)
    if np.isfinite(pe) and pe > mlr['bucket_summary_decision'].get('overall_ece10', np.inf) and pe > knr['bucket_summary_decision'].get('overall_ece10', np.inf):
        fails.append('Post-conformal bucket ECE worse than both ML-only and KNN-only')
    if not gate_diag.get('blend_beats_both', False): fails.append('Gate blend did not beat both experts on dev OOF')
    if post['decision_rows_point']['mae'] > mlr['decision_rows_point']['mae'] + 0.05: fails.append('Post-conformal decision MAE worse than ML-only by tolerance')
    if post['crossing_after_repair_decision'] > 0 or post['crossing_after_repair_all'] > 0: fails.append('Quantile crossing remains after repair')
    deployable = len(fails) == 0

    winner_mae = model_cmp.sort_values('decision_row_mae').iloc[0]['model'] if not model_cmp.empty else None
    winner_ece = model_cmp.sort_values('bucket_ece10').iloc[0]['model'] if not model_cmp.empty else None
    summary = {
        'data_summary': {'all_rows': int(len(feat_df)), 'decision_rows': int(len(decision)), 'train_rows': int(len(train_core)), 'dev_rows': int(len(dev)), 'test_rows': int(len(test))},
        'split_summary': cfg.split.__dict__, 'leakage_audit_pass': leak.passed, 'sanitization_summary': sanit,
        'feature_summary': {'feature_count': len(feat_cols), 'knn_feature_count': len(knn_cols)},
        'baseline_metrics': {'climo': res['climo baseline'], 'persistence': res['persistence baseline']},
        'ml_quantile_metrics': mlr, 'knn_metrics': knr, 'gate_metrics': gate_diag,
        'blend_pre_conformal_metrics': pre, 'blend_post_conformal_metrics': post,
        'decision_row_metrics': model_cmp[['model','decision_row_mae','avg_pinball','bucket_ece10']].to_dict(orient='records'),
        'interval_coverage': {'pre': pre['decision_rows_interval'], 'post': post['decision_rows_interval']},
        'bucket_calibration': {'pre': pre['bucket_summary_decision'], 'post': post['bucket_summary_decision']},
        'pit_summary': {'pre': pre['pit_summary_decision'], 'post': post['pit_summary_decision']},
        'deploy_recommendation': {'deployable': deployable, 'winner_by_mae': winner_mae, 'winner_by_bucket_ece': winner_ece, 'knn_helped': knr['decision_rows_point']['mae'] < mlr['decision_rows_point']['mae'], 'gate_helped': gate_diag.get('blend_beats_both', False), 'conformal_helped': abs(post['decision_rows_interval'].get('cov_90', np.nan) - 0.90) < abs(pre['decision_rows_interval'].get('cov_90', np.nan) - 0.90), 'top_failure_modes': fails[:3]},
        'optional_peak_delta_benchmark': 'Peak+delta benchmark unavailable',
    }

    model_cmp['deployable_flag'] = [deployable if m == 'blend post conformal' else False for m in model_cmp['model']]
    write_df(model_cmp, stages['09_reports'] / 'model_comparison.csv')
    write_json(stages['09_reports'] / 'summary.json', summary)
    (stages['09_reports'] / 'results_executive.md').write_text(executive_markdown(summary), encoding='utf-8')
    write_df(slice_metrics(test_dec, test_dec_conf, dfs['blend post conformal']['pmf_decision'], dfs['blend post conformal']['bucket_probs_decision'], dfs['blend post conformal']['realized_bucket_decision']), stages['09_reports'] / 'slice_metrics.csv')

    bdir = stages['10_bundle']
    with (bdir / 'quantile_models.pkl').open('wb') as f: pickle.dump(final_mpack, f)
    with (bdir / 'knn_model.pkl').open('wb') as f: pickle.dump(final_knn, f)
    with (bdir / 'gate_model.pkl').open('wb') as f: pickle.dump(gate, f)
    _yaml = {'buckets': [{'name': b.name, 'min_temp': b.min_temp, 'max_temp': b.max_temp} for b in buckets]}
    (bdir / 'bucket_config_snapshot.yaml').write_text(yaml.safe_dump(_yaml, sort_keys=False), encoding='utf-8')
    shutil.copyfile(Path(__file__).parent / 'live_infer.py', bdir / 'live_infer.py')
    (bdir / 'model_card.md').write_text('\n'.join(['# Model Card','', '- Target: KNYC same-day Tmax (F)','- Core model: LightGBM quantile grid + KNN analog + gate + rolling conformal',f'- Train/dev/test: {cfg.split.train_start}..{cfg.split.train_end} / {cfg.split.dev_start}..{cfg.split.dev_end} / {cfg.split.test_start}..{cfg.split.test_end}',f'- Leakage audit pass: {leak.passed}',f'- Deployable: {deployable}','']), encoding='utf-8')
    write_json(bdir / 'bundle_manifest.json', {'created_at_utc': _utc_now(), 'quantiles': cfg.model.quantiles, 'feature_cols': feat_cols, 'knn_feature_cols': knn_cols, 'deployable': deployable, 'leakage_audit_pass': leak.passed})
    write_manifest(stages['10_bundle'], {'stage': 'bundle', 'created_at_utc': _utc_now(), 'files': [p.name for p in bdir.iterdir()]})

    return {'summary': summary, 'model_comparison': model_cmp, 'output_dir': cfg.output_dir}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='KNYC quantile+KNN+gate+conformal pipeline')
    ap.add_argument('--config', default=None)
    ap.add_argument('--obs-csv', required=True)
    ap.add_argument('--truth-csv', required=True)
    ap.add_argument('--station-universe', required=True)
    ap.add_argument('--schema-profile', default=None)
    ap.add_argument('--bucket-config', default=None)
    ap.add_argument('--market-odds-file', default=None)
    ap.add_argument('--decision-stockholm-time', default='19:00')
    ap.add_argument('--train-end', default='2021-12-31')
    ap.add_argument('--dev-start', default='2022-01-01')
    ap.add_argument('--dev-end', default='2023-12-31')
    ap.add_argument('--test-start', default='2024-01-01')
    ap.add_argument('--test-end', default='2025-12-31')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--skip-sanitization', action='store_true')
    ap.add_argument('--force-rebuild-features', action='store_true')
    ap.add_argument('--k-neighbors', type=int, default=64)
    ap.add_argument('--conformal-window', type=int, default=365)
    ap.add_argument('--emit-global-diagnostics', action='store_true')
    ap.add_argument('--random-seed', type=int, default=42)
    ap.add_argument('--allow-debug-failure', action='store_true')
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    cfg = load_config(a.config, {
        'obs_csv': a.obs_csv, 'truth_csv': a.truth_csv, 'station_universe': a.station_universe,
        'schema_profile': a.schema_profile, 'bucket_config': a.bucket_config, 'market_odds_file': a.market_odds_file,
        'decision_stockholm_time': a.decision_stockholm_time, 'output_dir': a.output_dir, 'skip_sanitization': a.skip_sanitization,
        'force_rebuild_features': a.force_rebuild_features, 'emit_global_diagnostics': a.emit_global_diagnostics,
        'allow_debug_failure': a.allow_debug_failure,
        'split': {'train_end': a.train_end, 'dev_start': a.dev_start, 'dev_end': a.dev_end, 'test_start': a.test_start, 'test_end': a.test_end},
        'model': {'k_neighbors': a.k_neighbors, 'conformal_window': a.conformal_window, 'random_seed': a.random_seed},
    })
    out = run_pipeline(cfg)
    print(json.dumps(out['summary'], indent=2, default=str))


if __name__ == '__main__':
    main()

from pathlib import Path

import pandas as pd

from pipelines.quantile_knn_conformal.config import PipelineConfig
from pipelines.quantile_knn_conformal.run_pipeline import run_pipeline


def test_pipeline_smoke(tmp_path: Path):
    obs_path = tmp_path / 'obs.csv'
    truth_path = tmp_path / 'truth.csv'
    stations_path = tmp_path / 'stations.csv'
    out_dir = tmp_path / 'out'

    times = pd.date_range('2021-12-30 00:00:00+00:00', periods=240, freq='H')
    rows = []
    for t in times:
        for sid in ['KNYC:9:US','KJFK:9:US','KEWR:9:US','KTEB:9:US','KHPN:9:US','KISP:9:US','KBDR:9:US','KMMU:9:US','KLGA:9:US']:
            temp = 30 + (t.hour / 2.0)
            rows.append({
                'request_location_id': sid,
                'valid_time_utc': t.isoformat(),
                'temp': temp,
                'dew_pt': temp - 5,
                'rh': 60,
                'pressure': 30,
                'vis': 10,
                'wspd': 8,
                'wdir': 180,
                'gust': 12,
                'precip_hrly': 0,
                'clds': 'SCT',
                'wx_phrase': 'Partly Cloudy',
                'uv_index': 2,
                'uv_desc': 'Low',
                'wdir_cardinal': 'S',
            })
    pd.DataFrame(rows).to_csv(obs_path, index=False)

    truth_rows = []
    for d in pd.date_range('2021-12-30', periods=7, freq='D'):
        truth_rows.append({'station_id':'KNYC','date':d.date().isoformat(),'settled_tmax':40 + d.day % 5})
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)

    pd.DataFrame([
        {'request_location_id':'KNYC:9:US','role':'target'},
        {'request_location_id':'KJFK:9:US','role':'neighbor'},
        {'request_location_id':'KEWR:9:US','role':'neighbor'},
        {'request_location_id':'KTEB:9:US','role':'neighbor'},
        {'request_location_id':'KHPN:9:US','role':'neighbor'},
        {'request_location_id':'KISP:9:US','role':'neighbor'},
        {'request_location_id':'KBDR:9:US','role':'neighbor'},
        {'request_location_id':'KMMU:9:US','role':'neighbor'},
        {'request_location_id':'KLGA:9:US','role':'neighbor'},
    ]).to_csv(stations_path, index=False)

    cfg = PipelineConfig(
        obs_csv=str(obs_path),
        truth_csv=str(truth_path),
        station_universe=str(stations_path),
        output_dir=str(out_dir),
        skip_sanitization=True,
    )
    cfg.split.train_end = '2021-12-31'
    cfg.split.dev_start = '2022-01-01'
    cfg.split.dev_end = '2022-01-03'
    cfg.split.test_start = '2022-01-04'
    cfg.split.test_end = '2022-01-05'

    out = run_pipeline(cfg)
    assert 'summary' in out
    assert (out_dir / '09_reports' / 'summary.json').exists()

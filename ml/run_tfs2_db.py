import argparse
from pathlib import Path

from weather_ml.tfs2 import db


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert TFS2 sweep results into model_experiment.")
    parser.add_argument("sweep_json", help="Path to time_feature_sweep_v2.json")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    payload = db.load_sweep_summary(Path(args.sweep_json))
    sweep_id = payload.get("sweep_id")
    if not sweep_id:
        raise ValueError("Missing sweep_id in payload")
    engine = db.create_db_engine(args.db_url)
    db.upsert_model_experiments(engine, payload, sweep_id=sweep_id)
    print(f"Upserted TFS2 experiments for sweep {sweep_id}")


if __name__ == "__main__":
    main()

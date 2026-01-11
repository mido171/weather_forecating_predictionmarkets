# WX-201 — Python ML module scaffold (ml/) with pinned deps + test harness

## Objective
Create the Python project scaffold under `ml/` that will host all Epic #2 code.

## Requirements
- Create `ml/` folder with:
  - `pyproject.toml` (preferred) OR `requirements.txt` + `requirements.lock`
  - `src/weather_ml/` python package structure (empty modules ok)
  - `tests/` with pytest skeleton
  - `configs/` with example YAML config
- Add minimal README under `ml/` explaining how to:
  - create venv
  - install deps
  - run unit tests
  - run the training CLI
- Ensure repo root `mvn clean install` still succeeds (Python folder must not break Maven).

## Dependencies (minimum)
- pandas
- numpy
- sqlalchemy + mysql driver (e.g., pymysql) OR mysqlclient
- scikit-learn
- matplotlib
- pyyaml
- joblib
- pytest

## Acceptance Criteria
- [ ] `python -m pip install -r ...` succeeds (or `poetry install` / `uv sync`).
- [ ] `pytest` runs and passes with at least one placeholder test.
- [ ] A `python -m weather_ml.train --help` entrypoint exists (even if stubbed).
- [ ] Root `mvn clean install` still passes.


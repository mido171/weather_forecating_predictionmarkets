# Local environment setup

Create environments only when a task needs to run a component. Environment creation is not
part of the read-only startup protocol, and no project should reuse an environment whose
editable install points at an archived or different checkout.

## HKG research environment

```powershell
py -3.11 -m venv projects/hkg-tmax/.venv
projects/hkg-tmax/.venv/Scripts/python.exe -m pip install -e "projects/hkg-tmax[research,dev]"
projects/hkg-tmax/.venv/Scripts/python.exe -c "import hkg_tmax; print(hkg_tmax.__file__)"
```

The final path must resolve beneath the current repository. HKG path-only settings may be
read from the ignored `projects/hkg-tmax/.env`; provider and database credentials are not
loaded by the path layer.

## KLGA research environment

```powershell
py -3.11 -m venv projects/klga-tmax/.venv
projects/klga-tmax/.venv/Scripts/python.exe -m pip install -e projects/klga-tmax
projects/klga-tmax/.venv/Scripts/python.exe -c "import klga_tmax; print(klga_tmax.__file__)"
```

Do not trust an existing global editable install. Verify the module path before running a CLI.

## Shared Python packages

Install shared packages only into the environment that consumes them:

```powershell
python -m pip install -e packages/python/weather-live
python -m pip install -e packages/python/weather-ml
```

`weather-live` provides the `ml_live` import. Runtime outputs default to ignored `var/` state
or the explicitly configured `WEATHER_MARKETS_RUN_ROOT`.

## Verification

After installation, run component-scoped offline checks before any provider, database, server,
or trading command. A virtual environment and a successful import do not authorize external
effects.

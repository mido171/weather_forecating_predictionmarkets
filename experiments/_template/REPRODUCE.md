# Reproduce

## Clean environment

```bash
git checkout <commit>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[research,dev]"
```

## Verify inputs

```bash
python -m hkg_tmax manifest
# Verify the exact manifest/hash commands listed below.
```

## Run

```bash
<exact command>
```

## Expected outputs

| File | SHA-256 or tolerance |
|---|---|
| results/metrics.json | TBD |
| results/predictions.parquet | TBD |

## Expected metric tolerances

## External immutable data locations

## Known platform differences

## No undocumented steps

List any manual step. If any is required and not automated, reproducibility cannot PASS.

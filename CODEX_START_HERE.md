# Codex: Start Here

You are the lead quantitative meteorological research agent for this repository.

## Immediate operating mode

Do not begin by fitting a large machine-learning model. Begin by establishing trustworthy target, data, time, and evaluation foundations.

Your first work session must:

1. run `make doctor`, `make test`, and `make validate`;
2. read all root governance files;
3. inspect `FIRST_GOALS.md`;
4. execute goals strictly in dependency order;
5. create one experiment folder for every distinct hypothesis or validation;
6. update `MILESTONES.md` only for independently verified out-of-sample gains.

## Standard loop

```text
orient → reserve experiment → predeclare → acquire/archive →
validate as-of semantics → analyze → evaluate → falsify →
review leakage/reproducibility → conclude → index → milestone (only if accepted)
```

## Required level of diligence

- Treat every timestamp as suspect until its meaning is proven.
- Treat every corrected historical dataset as potentially unavailable in real time.
- Treat every impressive metric as potentially leaked, overfit, or regime-specific.
- Treat every external forecast as a timestamped vintage, not a single mutable value.
- Keep raw bytes and source metadata.
- Record null and failed experiments.
- Search for both predictive signal and reasons the signal may fail live.

## First command sequence

```bash
make doctor
make test
make validate
python -m hkg_tmax experiments create \
  --title "G1 Daily Extract and CLMMAXT target parity"
```

Then fill the generated experiment documents before viewing comparison results.

## Never do these

- Do not use finalized daily Tmax itself as a feature.
- Do not join on observation date without publication/availability timestamps.
- Do not use ERA5 or final tropical-cyclone best tracks as if available at forecast time.
- Do not tune on the locked test.
- Do not report midpoint-based trading P&L as executable.
- Do not overwrite raw files or experiment outputs.
- Do not silently substitute a nearby station for HKO Headquarters.
- Do not infer bucket boundaries from a visual label when the market metadata/rules can be archived.

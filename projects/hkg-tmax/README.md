# HKG Tmax Elite Codex Research System

This repository is a complete starting environment for a rigorous, Codex-driven research program focused on forecasting the contract-authoritative Hong Kong daily maximum temperature.

It is designed around four truths:

1. **Settlement semantics are part of the prediction problem.**
2. **Historical data is not automatically point-in-time data.**
3. **Accuracy improvements must survive locked out-of-sample tests.**
4. **A weather edge becomes a trading edge only after execution costs and fill realism.**

## Start here

```bash
cp .env.example .env
bash scripts/bootstrap.sh
make doctor
make test
make validate
python -m hkg_tmax experiments create --title "Verify Daily Extract versus CLMMAXT parity"
```

Then read, in order:

1. `CODEX_START_HERE.md`
2. `FIRST_GOALS.md`
3. `AGENTS.md`
4. `docs/PROJECT_STRUCTURE_AND_CODE_MAP.md`
5. `docs/01_RESEARCH_CHARTER.md`
6. `docs/02_TARGET_AND_SETTLEMENT.md`
7. `docs/03_ASOF_AND_LEAKAGE.md`

## Repository map

```text
.
├── AGENTS.md                     Codex operating constitution
├── FIRST_GOALS.md                Ordered initial goal program
├── MILESTONES.md                 Human-facing accepted findings dashboard
├── EXPERIMENT_INDEX.md           Registry summary
├── .codex/agents/                Specialist Codex subagents
├── .agents/skills/               Repeatable Codex skills
├── code/
│   ├── src/                       Python packages: hkg_tmax and hkg_tmax_db
│   └── tests/                     Pytest suite
├── config/                       Source, target, horizon, split, station config
├── docs/                         Research and operational specifications
├── data/                         Raw/bronze/silver/gold schemas and archives
├── experiments/                  One immutable folder per experiment
├── scripts/                      Bootstrap, archive, and validation commands
├── migrations/                   SQL migrations
└── tasks/                        A-to-Z task packages and completion records
```

## What this bootstrap already implements

- repository and experiment governance;
- Codex subagents and skills;
- immutable raw snapshot writer with hashes and sidecars;
- source catalog and fetch CLI;
- market-rules snapshot support;
- experiment-folder generator;
- milestone renderer;
- strict local-time and information-cutoff helpers;
- generic contract-bucket mapping with boundary tests;
- source/config validation;
- test suite and CI workflow;
- detailed data-acquisition and model-research roadmap.

## What must be proven before modelling

The market rules name HKO’s Daily Extract and the field `Absolute Daily Max (deg. C)`. The official HKO climate API also offers a `CLMMAXT` daily maximum series for station `HKO`. They are highly plausible matches, but **the repository intentionally does not assume perfect parity**. Goal G1 archives both and proves date-by-date equivalence, including publication timing and revisions. Until it passes, `CLMMAXT` is a candidate training label, not canonical settlement truth.

## No profit guarantee

This repository is designed to maximize research quality, not to promise profitability. Weather uncertainty, market adaptation, source changes, fees, spreads, latency, fills, and model decay can erase apparent edge. Production trading remains disabled until all gates in `docs/07_PRODUCTION_GATE.md` pass.

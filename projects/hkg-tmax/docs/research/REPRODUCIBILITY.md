# Computational Reproducibility

## Minimum artifact set

An accepted experiment must preserve:

- source/file manifests and hashes;
- code commit;
- dirty patch or clean-state confirmation;
- config files and hashes;
- dependency snapshot;
- platform/Python info;
- seeds;
- command;
- logs;
- row-level predictions;
- metrics;
- plots;
- conclusion;
- reviewer sign-offs.

## Environment

Bootstrap supports `venv` and editable installation. For stronger freezing, Codex should later add a generated lock file on the target platform and container definition after dependencies are selected.

Do not fabricate a lock file manually. Generate it with the chosen resolver and archive resolver version.

## Deterministic data build

Derived dataset IDs should hash:

- sorted raw input hashes;
- parser/transform code version;
- config hash;
- schema version.

The same inputs must yield the same output hash where deterministic formats permit.

## Git

Before accepted experiments:

```bash
git init
git add .
git commit -m "Bootstrap HKG Tmax research system"
```

For every run capture:

```bash
git rev-parse HEAD
git status --porcelain
git diff --binary
python --version
python -m pip freeze
```

## Randomness

- central seed in run config;
- per-component derived seeds;
- multiple seeds for stochastic challengers;
- report distribution, not best seed.

## Rerun review

The reproducibility reviewer should use a clean checkout and only documented commands. Hidden caches or manually copied files are failures.

## Artifact retention

Small metadata/docs/metrics belong in Git. Large raw/model/output files should live in durable storage with manifests and checksums. A path without a hash is not a reproducible reference.

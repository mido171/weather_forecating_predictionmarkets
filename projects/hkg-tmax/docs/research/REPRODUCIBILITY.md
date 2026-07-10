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

This project already lives inside the standalone `weather_data_extraction`
repository. Never initialize a nested repository and never stage with
`git add .` or `git add -A`. Stage only reviewed, explicit experiment/code/test
paths after inspecting the scoped diff and untracked files.

Before accepting an experiment, verify root, branch, remote, tracked-only
status, staged names, staged statistics, and the complete staged diff according
to `AGENTS.md`.

For every run capture:

```bash
git rev-parse HEAD
git -c core.fsmonitor=false status --porcelain --untracked-files=no
python --version
python -m pip freeze
```

If the run starts from a dirty tree, archive a binary diff limited to the
declared source/config/test inputs. Record that exact path list and patch hash
in the run manifest; do not trigger a root-wide untracked scan.

## Randomness

- central seed in run config;
- per-component derived seeds;
- multiple seeds for stochastic challengers;
- report distribution, not best seed.

## Rerun review

The reproducibility reviewer should use a clean checkout and only documented commands. Hidden caches or manually copied files are failures.

## Artifact retention

Small metadata/docs/metrics belong in Git. Large raw/model/output files should live in durable storage with manifests and checksums. A path without a hash is not a reproducible reference.

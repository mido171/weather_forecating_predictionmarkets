# Script registry and rules

This directory intentionally remains flat for compatibility: many tests and
historical experiment manifests import scripts by filename. A historical
script is therefore not moved merely to improve appearance when that move would
break reproducibility imports. `REGISTRY.csv` makes lifecycle explicit:

- `active_operator` and `active_research` are reviewed current entry points;
- `retained_reproduction` exists only to reproduce or inspect archived evidence
  and is not current research authority;
- generic `operator_workflow`, `research_workflow`, and `maintained_utility`
  entries have not received a stronger lifecycle override.

Retained reproduction scripts write mutable outputs through `ProjectPaths` to
the configured external data/run roots. Their evidence remains under
`docs/evidence`; rerunning one does not promote its conclusion or authorize
locked-test access. New reusable logic belongs in `src/`; a script should be a
thin, bounded entry point.

`REGISTRY.csv` is the generated quick index. Refresh it with
`python scripts/build_script_registry.py`; the generator inspects only immediate
Python files and caps text reads per file. The `project_paths_detected` column
is an inventory signal, not proof that every path in a script is correct.

Scripts are classified by filename:

- `check_`, `audit_`, `profile_`, `inspect_`: offline/read-only diagnostics
- `build_`, `generate_`, `normalize_`: derived-output builders
- `run_`: experiment or bounded workflow entry points
- `backfill_`, `fetch_`, `download_`: network/data operations; require explicit
  execution acknowledgement and hard budgets
- `install_`, `start_`, `monitor_`: operations; dry-run and finite by default

Every new script must have a focused test, `--help`, safe defaults, a finite
runtime, worker count one, and no network side effect without `--execute`.

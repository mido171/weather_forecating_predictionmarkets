# Background-run ledger contract

Every background, network, backfill, training, render, or server process needs a
JSON Lines entry under `${HKG_TMAX_RUN_ROOT}/ledger/runs.jsonl` containing:

- run ID, owner, purpose, classification, and dry-run/live mode;
- exact executable, arguments, working directory, PID, and child PIDs;
- provider/source/date scope and request/byte/runtime/retry budgets;
- worker and BLAS thread settings;
- start/end timestamps, exit status, stop reason, and exact stop command;
- log path, manifest path, output paths, database row counts, and Git commit.

Never infer ownership from an executable name. Revalidate command line, parent
PID, and listening port before stopping a process.

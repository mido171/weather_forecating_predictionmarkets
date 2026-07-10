# Local runtime state

Everything below this directory is ignored except this file. Store only lightweight local
PID records, run metadata, logs, caches, and temporary state here. Bulk datasets and durable
run artifacts belong in the configured external data/run store.

Each background process record under `var/run/` must include PID and children, exact redacted
command, working directory, owner/task, start time, port, log path, health check, and exact
stop command.

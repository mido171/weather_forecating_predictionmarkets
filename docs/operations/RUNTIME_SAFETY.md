# Runtime and Resource Safety

## Process ownership contract

Before acting on a process, record its executable, full redacted command line, parent and
children, creation time, working directory, port, owner/task, and log. Re-query immediately
before stopping it. Stop only the verified process tree, children first.

Never use name-wide termination such as killing every Python, Java, Node, or browser process.
PIDs in documentation are historical evidence and must never be reused without live proof.

## Background process record

Each background process writes ignored `var/run/<service>.json` containing:

```json
{
  "pid": 0,
  "child_pids": [],
  "command": "redacted exact command",
  "working_directory": "absolute path",
  "owner": "task or operator",
  "started_at": "RFC3339 timestamp",
  "port": null,
  "log_path": "absolute path outside Git",
  "health_check": "bounded command",
  "stop_command": "exact scoped command"
}
```

## Live-run budget

Collectors and backfills require provider, endpoint, station/location, date/cutoff, maximum
requests, bytes, rows, retries, concurrency, runtime, destination, idempotency key, dry-run,
and stop condition. Default concurrency is one and may not exceed two without approval.

## Startup prohibition

Repository orientation and verification do not start services, collectors, schedulers,
WebSockets, database migrations, backfills, or trading. Startup listeners must be fail-closed
unless an explicitly selected runtime profile enables them.

# Repository Doctor

`doctor.py` is a bounded, read-only policy check. It uses the Python standard library and
read-only Git commands. It never follows reparse points or starts product code.

```powershell
python tools/repo/doctor.py
python tools/repo/doctor.py --strict
python tools/repo/doctor.py --scope projects/hkg-tmax
python tools/repo/doctor.py --json
python -m unittest discover -s tools/repo/tests -p "test_*.py"
```

Checks cover standalone/nested Git boundaries, junctions and symlinks, required root docs,
tracked runtime files, files over the configured size limit, high-confidence secret shapes,
credential-like literal assignments, stale workspace paths, unsafe enabled defaults,
unbounded model jobs, high concurrency, startup listeners, and infinite loops that require
manual cancellation review.

Exit codes:

- `0`: no errors; warnings are informational unless `--strict` is used.
- `1`: policy errors, or warnings under `--strict`.
- `2`: invalid arguments or a check could not run reliably.

Findings never print credential values. Fix the underlying issue instead of globally
suppressing a check. For an intentionally guarded startup/listener/concurrency construct, a
same-line or nearby `repo-doctor: allow-unsafe-default` marker is accepted only after the
guard, cancellation behavior, and scoped resource budget have been reviewed.

# Results

## Run status

Completed.

## Metrics

- poll iterations completed: `6`
- poll snapshot count: `6`
- row count: `17`
- evidence counts: `{"ARCHIVE_FIRST_OBSERVED_NOT_PROVIDER_FIRST": 17}`
- provider first-publication candidate count: `0`
- provider first publication proven: `false`
- revision count: `0`
- watched candidate dates present: `[]`
- watched candidate dates missing: `["2026-06-18"]`
- final catalog retrieved_at: `2026-06-18T20:57:54.578715Z`
- final catalog hash:
  `f80772b68545c56e6842c34998696fd11b7b9a80c0088bb1f6e4da65102616eb`
- final monthly retrieved_at: `2026-06-18T20:57:57.421862Z`
- final monthly hash:
  `c50910ab74e2ba8bff1f661fb1ae663d15b128dae0dfb4ed97c0e40c97bcbefc`

## Archive verification

PASS after direct sidecar/hash verification:

- raw snapshots: `12`
- metadata sidecars: `12`
- unique raw paths: `true`
- SHA-256 recomputation matched sidecar `content_sha256`
- sidecars contained source ID, retrieval timestamp, storage schema version,
  HTTP 200 status, requested URL, final URL, request headers, and response
  headers

## Validation

Pre-poll validation:

```powershell
.\.venv\Scripts\python.exe -m hkg_tmax validate all
```

Final gates:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m hkg_tmax validate all
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
```

PASS. `pytest` reported 59 passed. Validation passed with the expected G1/G2
gate warnings only. Ruff and MyPy passed.

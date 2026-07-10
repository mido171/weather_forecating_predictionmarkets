# Full Tactical Backfill Stop Marker

Stopped at: 2026-06-25 09:43:11 UTC / 2026-06-25 11:43:11 Europe/Stockholm

Reason: GribStream returned HTTP 429 during the full tactical H24N Tmax backfill. The process entered retry sleep and was manually stopped as requested.

Completed before stop:

- Planned chunks: 1,079
- Completed chunks: 217
- Rows returned/upserted by completed chunks: 788,147
- Estimated credits consumed by completed chunks: 769,286
- HTTP errors before the stop point: none until the 429 response

Completed dataset coverage:

- gfs: 138 chunks, 575,004 rows, 622,921 estimated credits, 2021-03-22T00:00:00Z through 2026-06-22T00:00:00Z
- gefsatmosmean: 68 chunks, 200,436 rows, 133,624 estimated credits, 2020-10-01T18:00:00Z through 2026-06-21T18:00:00Z
- gefsatmos: 11 chunks, 12,707 rows, 12,741 estimated credits, 2020-10-01T18:00:00Z through 2020-11-24T18:00:00Z

429 event details:

- Request SHA256: b6bbba1718059cc401b4ba71956c781f5a664ae371f154d9cc643c5039fa04b6
- Retry-After header: 51407 seconds
- Script retry sleep logged: 1800 seconds

Process handling:

- Matched backfill Python processes were stopped manually after the 429.
- Remaining matched backfill processes after stop: 0

Primary files:

- Progress: experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/progress.json
- Batch results: experiments/0214_tactical_h24n_gribstream_backfill/full_tactical_backfill_ok_tmax/batch_results.csv
- API events: experiments/0214_tactical_h24n_gribstream_backfill/logs/gribstream_full_tactical_backfill_ok_tmax_api_events.jsonl
- Raw data root: data/_pipeline_internal/raw/gribstream_tactical_full_tactical_backfill_ok_tmax

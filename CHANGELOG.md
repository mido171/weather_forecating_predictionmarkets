# Changelog

## G1 target-station checkpoint — 2026-06-18

- Added HKO Daily Extract backing-payload parsing for `Absolute Daily Max (deg. C)`.
- Fixed CLMMAXT parsing for real HKO bilingual CSV headers and footer rows.
- Added fail-closed target adapter tests for missing source/field/value, ambiguous
  date, unsupported precision, station mismatch, and source failure.
- Generated a May 2026 latest-payload parity sample: 31/31 Daily Extract rows
  matched CLMMAXT HKO.
- Marked EXP-0002 `BLOCKED` pending first-publication Daily Extract evidence;
  no predictive modelling or Polymarket backtesting was run.
- Added source contracts for Daily Extract, CLMMAXT HKO, and HKO station metadata.
- Added EXP-0003 Daily Extract polling infrastructure and a first-observation
  ledger for June 2026; it is accepted as infrastructure, while G1 remains
  blocked pending provider first-publication evidence.
- Added EXP-0004 bounded Daily Extract polling with explicit metrics output and
  watched-date provider-first candidate gating; G1 remains blocked until actual
  provider first-publication evidence is observed.
- Added EXP-0005 active Daily Extract watch for `2026-06-18` and tightened
  provider-first candidate gating to require active absent-before-present raw
  snapshots before candidate status.
- Added EXP-0006 continuation polling for `2026-06-18`, per-iteration poll
  snapshot metrics, and explicit bounded fetch retries for transient provider
  disconnects.
- Added EXP-0007 second continuation polling for `2026-06-18`; the Daily
  Extract monthly payload remained unchanged and the watched row was still
  absent.
- Added EXP-0008 third continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:20:46Z`.
- Added EXP-0009 fourth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:30:12Z`.
- Added EXP-0010 fifth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:38:03Z`.
- Added EXP-0011 sixth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:45:51Z`.
- Added EXP-0012 seventh continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T18:55:41Z`.
- Added EXP-0013 eighth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:03:26Z`.
- Added EXP-0014 ninth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:10:57Z`.
- Added EXP-0015 tenth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:18:33Z`.
- Added EXP-0016 eleventh continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:27:34Z`.
- Added EXP-0017 twelfth continuation polling for `2026-06-18`; six more active
  poll iterations completed, archive sidecars were verified, and the watched
  row was still absent through `2026-06-18T19:35:51Z`.
- Added EXP-0018 thirteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:42:55Z`.
- Added EXP-0019 fourteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:50:50Z`.
- Added EXP-0020 fifteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T19:58:46Z`.
- Added EXP-0021 sixteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:06:57Z`.
- Added EXP-0022 seventeenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:15:05Z`.
- Added EXP-0023 eighteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:24:33Z`.
- Added EXP-0024 nineteenth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:33:50Z`.
- Added EXP-0025 twentieth continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:42:02Z`.
- Added EXP-0026 twenty-first continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:50:08Z`.
- Added EXP-0027 twenty-second continuation polling for `2026-06-18`; six more
  active poll iterations completed, archive sidecars were verified, and the
  watched row was still absent through `2026-06-18T20:57:57Z`.

## G0 smoke test — 2026-06-18

- Archived every `bootstrap_now` source twice and documented EXP-0001.
- Generated `reports/source_inventory.md` and refreshed `MANIFEST.json`.
- Added fetch failure-path tests for HTTP errors and empty payloads.
- Added malformed HKO climate CSV parser coverage.
- Made experiment creation tests independent of the live registry counter.
- Fixed `scripts/bootstrap.ps1` so native command failures stop the bootstrap.
- Added compatibility docs for requested `HKG_TMAX_FIRST_GOALS.md`,
  `docs/00_PROJECT_OVERVIEW.md`, and `docs/06_LEAKAGE_CONTROL.md` paths.

## 0.1.0 — 2026-06-18

- Created complete Codex research bootstrap.
- Added target-parity and rules-change gates.
- Added point-in-time timestamp model and leakage controls.
- Added immutable raw archive and source catalog scaffolding.
- Added experiment ledger, milestone renderer, specialist agents, and skills.
- Added initial goal program from target verification through production eligibility.
- Added tests for bucket mapping, as-of availability, immutability, and config integrity.

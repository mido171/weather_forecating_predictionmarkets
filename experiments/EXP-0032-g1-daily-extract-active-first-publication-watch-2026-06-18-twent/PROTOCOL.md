# Protocol

Superseded. Do not rerun the old polling command.

The replacement protocol is operational data acquisition:

- Daily Extract: at most one successful fetch per Asia/Hong_Kong local day at
  09:00, with one retry six hours later only after failure.
- Unchanged hashes: deduplicated raw object, append one retrieval-ledger row.
- Experiments: no experiment folder, test suite run, or commit for routine
  unchanged retrievals.
- Scope: weather data acquisition only; no Polymarket, settlement parity,
  modelling, or locked-test work.

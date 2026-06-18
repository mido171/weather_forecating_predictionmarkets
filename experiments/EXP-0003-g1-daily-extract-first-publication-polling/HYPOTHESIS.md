# Hypothesis

## Mechanism

HKO Daily Extract values are exposed through JSON-text backing payloads under
`/cis/dailyExtract/`. If the repository polls and archives those payloads
immutably, then each target row can be assigned an archive-observed first-seen
timestamp, raw hash, completeness state, and later revision history.

## Exact Prediction

- A single command can archive the Daily Extract coverage catalog and one
  monthly backing payload.
- The command can parse all available target-station rows for the requested
  month using the EXP-0002 parser.
- The command can build a ledger with one row per local date and raw-hash
  provenance for first archive observation and latest observation.
- Rows that already existed before polling began will be marked as archive
  first-observed evidence only, not true provider first-publication proof.
- No predictive model, machine learning, market price history, order book,
  trade, liquidity, execution, or replay artifact is created.

## Null Hypothesis

The repository cannot reliably poll and ledger Daily Extract target rows without
ambiguous dates, missing raw provenance, unsupported precision, unsafe source
state, or misleading first-publication claims.

## Falsification

- A raw payload is parsed before being archived.
- A target row appears without source ID, raw hash, retrieved-at timestamp, or
  quality state.
- Existing historical rows are labelled as provider first-published without
  evidence.
- The command cannot rerun deterministically from archived sidecars.
- Parser or target adapter safety tests fail.

## Novelty And Prior Evidence

EXP-0002 found that the Daily Extract HTML shell references root-relative
backing payloads and that latest May 2026 Daily Extract matched CLMMAXT HKO
31/31. It did not create live polling or first-observation ledgers.

## Leakage Risks Anticipated

- Treating first archive observation as true first publication without active
  near-publication polling.
- Replacing earlier raw payloads with later payloads.
- Letting latest corrected target values become training labels before G1
  passes.
- Treating CLMMAXT as canonical when it remains a proxy.

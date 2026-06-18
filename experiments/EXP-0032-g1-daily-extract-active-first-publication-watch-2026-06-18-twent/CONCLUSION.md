# Conclusion

## Decision

`SUPERSEDED`

## Rationale

The Daily Extract polling-loop family is closed by direct user instruction. This
folder preserves the interrupted EXP-0032 evidence but must not be used as an
accepted experiment, milestone, or reason to continue polling.

## Replacement policy

Daily Extract collection is moved to acquisition operations:

- at most one successful fetch per Asia/Hong_Kong local day at 09:00;
- one retry six hours later only after a failed request;
- unchanged payload hashes are deduplicated;
- every retrieval attempt is recorded in an append-only ledger;
- no experiment folder is created for routine unchanged payloads.

## Explicit non-claims

This does not prove settlement parity, does not unblock modelling, and does not
authorize Polymarket work.

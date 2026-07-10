---
name: verify-target-parity
description: Prove the exact HKO/Polymarket settlement target, first-publication semantics, CLMMAXT parity, and bucket winner mapping. Use before any target-dependent modelling.
---

1. Archive exact event JSON and rules text.
2. Extract source, field, date, timezone, precision, revision language, and outcomes.
3. Archive first-published Daily Extract values where recoverable.
4. Retrieve CLMMAXT station HKO and latest Daily Extract values.
5. Build the parity table and inspect every mismatch.
6. Apply rules-derived half-open bucket intervals and compare to actual resolved winners.
7. Add real-event regression fixtures.
8. Fail closed for unknown rules hash, missing field, overlaps, gaps, or ambiguous date.
9. Do not call CLMMAXT canonical until parity passes.

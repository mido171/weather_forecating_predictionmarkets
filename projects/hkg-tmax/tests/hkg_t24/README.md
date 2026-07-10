# HKG T24 Full Strategy Tests

All new tests for the HKG T+24 / H24N full strategy implementation must live here or in a clearly named subfolder here.

The four Jira packets require unit, integration, smoke, leakage, and temporal-integrity tests. Keep those tests under:

```text
tests/hkg_t24/
```

Do not scatter new strategy tests across unrelated legacy experiment test files unless a legacy module is directly changed.

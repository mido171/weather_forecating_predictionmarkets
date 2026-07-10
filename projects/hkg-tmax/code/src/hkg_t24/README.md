# HKG T24 Full Strategy Implementation Package

This package is the dedicated implementation home for the HKG T+24 / H24N full strategy contract.

All new strategy implementation code for the four Jira packets under:

```text
documentation/strategy_implementation_documentation/actual_strategy_implementation_contract/jira_breakdow/
```

must live under:

```text
code/src/hkg_t24/
```

This preserves the contract package name `hkg_t24` while following this repository's actual Python package root, `code/src`.

## Subpackage Ownership

```text
audit/       Source registries, schema checks, leakage audit events, provenance.
db/          Database connection, migrations, DDL helpers, SQL execution boundaries.
features/    H24N snapshots, official/target/NWP/proxy feature builders, matrices.
models/      Expert models, OOF generation, routers, specialists, distribution.
validation/  Scoreboards, leakage tests, negative controls, sealed validation.
live/        Live prediction, replay prediction, post-settlement scoring.
orchestration/
             End-to-end phase runners and full-pipeline coordination.
artifacts/   Candidate freeze and artifact manifest helpers.
utils/       Narrow shared helpers with no domain-policy ownership.
```

Do not place full-strategy implementation code in ad hoc scripts, root files, notebooks, reports, or Jira folders. Scripts may call this package, but they must not become the authoritative implementation.

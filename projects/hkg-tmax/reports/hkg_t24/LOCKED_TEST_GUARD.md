# Locked-Test Guard

Ordinary HKG T-24 research commands must reject target dates greater than or
equal to `2025-01-01`. Existing archived rows for 2025-2026
may remain on disk, but research code must not compute losses, select features,
tune models, or inspect failure cases on those rows.

The guard is implemented in `hkg_tmax.hkg_t24.guard` and covered by unit tests.
Any future explicit unlock must be audited separately and was not invoked for
this goal.

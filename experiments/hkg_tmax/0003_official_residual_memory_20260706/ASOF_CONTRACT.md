# As-Of Contract

For prediction target date `T` at cutoff `c`, a residual-memory source row for prior date `d` is eligible only when `d <= T-2` and the prior official anchor for `d` was selected using the same cutoff profile `c`.

The selected target-day official anchor itself must be the latest eligible Info.gov local forecast row with issue time at or before the cutoff.

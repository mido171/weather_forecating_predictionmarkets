# As-Of Contract

The decision cutoff is T-1 23:59 HKT. A forecast row is eligible only if `issue_at_utc <= cutoff_at_utc`. An hourly reading is eligible only if both `dispatch_at_utc <= cutoff_at_utc` and `observation_at_utc <= cutoff_at_utc`.

The experiment uses only the 24-hour observation window ending at the cutoff. It does not use target-day observations after the cutoff and does not read `sealed_confirmation` labels.

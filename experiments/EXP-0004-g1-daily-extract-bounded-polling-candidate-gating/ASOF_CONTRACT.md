# As-Of Contract

EXP-0004 does not create forecast rows. It creates publication-evidence rows.

Provider-first-publication candidate status is allowed only when:

```text
first_archive_retrieved_at >= active_polling_start_at
and local_date in watched_candidate_dates
and revision_observed == false
```

All timestamps must be timezone-aware. The output remains target-publication
evidence only; it is not an operational model feature.

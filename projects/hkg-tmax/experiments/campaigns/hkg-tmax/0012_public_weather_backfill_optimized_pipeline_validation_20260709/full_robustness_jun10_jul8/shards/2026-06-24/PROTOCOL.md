# Protocol

Process one UTC day at a time. For each source issue, download only the required raw payload,
validate it, normalize scalar station and area features, write source metadata and features to
Postgres, then delete raw bytes before moving to the next item. Every feature must join to a
source issue with `available_at_utc`.

Optimized mode uses bounded fetch/normalize workers, keeps DB writes serialized in the main
process, and deletes each raw payload only after the DB commit or recorded failure handling.

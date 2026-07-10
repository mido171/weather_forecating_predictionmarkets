# HKG-T24-001 Source Registry Migration

## Status

PASS

## Final Shape

The migration uses `source_code` plus explicit `strict_allowed`, `proxy_allowed`, `shadow_allowed`, `blocked`, `live_only`, and `support_only` columns. New code does not read or populate deprecated `strict_status`.

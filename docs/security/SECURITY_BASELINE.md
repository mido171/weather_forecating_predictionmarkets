# Security Baseline

## Secrets

- Secrets come from environment variables or an approved secret store.
- `.env`, `.secrets`, and local secret directories are ignored.
- Example configuration contains placeholders, never usable credentials.
- Do not print secrets in commands, logs, reports, screenshots, patches, or error messages.
- Recovery bundles and historical repositories are sensitive because removed credentials may
  remain in Git history; restrict their ACL and retention.

If a tracked credential is discovered, do not repeat it. Neutralize the current configuration,
rotate/revoke it at the provider, assess history separately, and block publishing until the
current tree is safe.

## Fail-closed production behavior

These capabilities default to disabled and require explicit runtime profiles and authorization:

- provider ingestion and high-volume polling;
- historical backfills;
- scheduled collectors and startup listeners;
- production account authentication;
- WebSocket subscriptions with account state;
- reconciliation or database mutation on startup;
- order construction, signing, placement, cancellation, or trading.

Testing production-capable code does not authorize a production connection or order.

## Cost and abuse controls

Public or provider-facing operations require authentication/authorization where appropriate,
rate and concurrency limits, bounded retries, timeouts, payload/row limits, retention limits,
idempotency, monitoring, and a cost/quota budget. Logs must not amplify request volume or
contain credentials or sensitive account data.

## Repository gate

`python tools/repo/doctor.py --strict` checks tracked content for high-confidence credential
shapes, credential-like literal assignments, unsafe runtime defaults, stale absolute paths,
large files, and runtime output. It supplements provider-side rotation and human review; it
is not proof that history or external systems contain no secrets.

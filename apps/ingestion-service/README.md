# Ingestion service

This service is intentionally inert and local-only unless an operator opts in to a specific ingestion path.

## Safe default contract

- The default Spring profile is `local`, backed by an in-memory H2 database.
- The HTTP server binds to `127.0.0.1`.
- CLI settlement ingestion, GribStream variable ingestion, Weather.com ingestion, all pilot jobs, and the KLGA IEM MOS backfill are disabled.
- Network helper scripts default to one worker and one retry. Higher values require an explicit CLI choice.
- Backfill and ingestion thread pools default to one worker, checkpoint resets are disabled, and Weather.com runs have bounded location, date, and task budgets.
- Proxy use is disabled and no proxy credential is stored in the repository.
- Generated logs and bulk training CSVs do not belong under `src/main/resources`; write runtime products beneath the repository `var/` tree.

## Administrative HTTP API

The Weather.com and pilot administrative controllers are not registered unless all of the following are deliberate:

1. Set `INGESTION_ADMIN_API_ENABLED=true`.
2. Set a non-empty `INGESTION_LOCAL_CONTROL_TOKEN`.
3. Send that value in the `X-Local-Control-Token` request header.

With the API disabled, its routes return `404`. With it enabled but without the correct token, they return `401`. The server remains loopback-only unless `INGESTION_SERVER_ADDRESS` is explicitly changed.

## External credentials and opt-in switches

Never put credentials in tracked YAML. Use environment variables:

The proxy credential that previously appeared in tracked configuration must be treated as compromised and rotated at the provider. Replacing it in the working tree prevents future use but does not erase it from existing Git history.

| Capability | Required opt-in |
| --- | --- |
| MySQL profile | `WEATHER_DB_URL`, `WEATHER_DB_USERNAME`, `WEATHER_DB_PASSWORD` |
| KLGA IEM MOS profile | `KLGA_IEM_MOS_JDBC_URL`, `KLGA_IEM_MOS_DB_USER`, `KLGA_IEM_MOS_DB_PASSWORD`, plus an explicit `iem-mos.enabled=true` override |
| Evomi proxy | `EVOMI_PROXY_ENABLED=true` and `EVOMI_PROXY_CREDENTIAL` |
| GribStream | `GRIBSTREAM_API_TOKEN` plus an explicit runner/ingestion enable flag |
| Weather.com | API key plus an explicit `weathercom.ingestion.enabled=true` override |
| Any runner or backfill | Its own explicit `enabled=true` override and a narrowly reviewed date/station scope |

Before enabling a runner, verify its date range, station count, thread count, retry count, output directory, and API quota. Start with a one-day, one-station dry run where the provider supports it.

## Verification

From the repository root:

```powershell
mvn -pl apps/ingestion-service -am test
```

The application-context test asserts that administrative controllers are absent under the default configuration.

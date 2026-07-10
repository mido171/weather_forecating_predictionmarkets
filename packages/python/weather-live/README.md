# Weather Live

Shared, importable live-weather adapters used by bounded operator tools. The installable
package is `ml_live`; the distribution name is `weather-live`.

```powershell
python -m pip install -e packages/python/weather-live
```

The compatibility namespace `ml_live.python` remains available for retained callers, while
new code should import `ml_live.fetch`, `ml_live.db`, or `ml_live.runtime`.

Runtime output defaults under the repository's ignored `var/` directory. Override it with
`WEATHER_MARKETS_RUN_ROOT`. Override the non-secret KMIA configuration example with
`WEATHER_MARKETS_LIVE_CONFIG`. Importing this package performs no provider call or database
mutation.

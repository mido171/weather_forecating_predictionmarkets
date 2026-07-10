# GribStream Credential Location

The local GribStream API token is stored in:

```text
secrets/local/gribstream.env
```

The file defines:

```text
GRIBSTREAM_API_KEY=<local token value>
```

Do not copy the token value into documentation, task artifacts, experiment logs, shell transcripts, screenshots, or commits.

## How Agents Should Use It

For authenticated GribStream work, load `GRIBSTREAM_API_KEY` from `secrets/local/gribstream.env`.

T07+ acquisition code treats `secrets/local/gribstream.env` as the project source of truth. It must not prefer a stale legacy `GRIBSTREAM_API_TOKEN` process environment variable over this file.

PowerShell process-local load example:

```powershell
Get-Content .\secrets\local\gribstream.env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
```

The key is needed for T03 authenticated catalog/selector/coverage probes and for T06+ acquisition tasks.

If this file is missing, T03 can still complete public documentation work, but authenticated probes must be marked blocked. T06 and later GribStream acquisition tasks must not run live API requests without this variable.

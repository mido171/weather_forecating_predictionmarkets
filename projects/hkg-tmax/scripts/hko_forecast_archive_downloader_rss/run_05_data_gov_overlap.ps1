$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DataRoot = Join-Path $ProjectRoot "data\datasets"

& $Python (Join-Path $ScriptDir "hko_archive.py") data-gov `
  --start 2020-01-01 `
  --end 2026-06-19 `
  --data-root $DataRoot `
  --delay-seconds 0.75

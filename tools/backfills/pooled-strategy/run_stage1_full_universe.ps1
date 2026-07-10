param(
    [string]$Stations = "all",
    [string]$DataRoot = "D:\Ahmed\data\sqlite\pooled_strategy",
    [switch]$DownloadMissingKalshi,
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$Python = "python"

$ArgsList = @(
    (Join-Path $RepoRoot "tools\pooled_strategy\stage1_backfill.py"),
    "--stations", $Stations,
    "--data-root", $DataRoot,
    "--log-level", $LogLevel
)

if ($DownloadMissingKalshi) {
    $ArgsList += "--download-missing-kalshi"
}

Write-Host "Running pooled stage-1 backfill..."
Write-Host ("Repo root: {0}" -f $RepoRoot)
Write-Host ("Stations:  {0}" -f $Stations)
Write-Host ("Data root: {0}" -f $DataRoot)
Write-Host ("WU key present: {0}" -f [bool]$env:WEATHERCOM_API_KEY)

Push-Location $RepoRoot
try {
    & $Python @ArgsList
}
finally {
    Pop-Location
}

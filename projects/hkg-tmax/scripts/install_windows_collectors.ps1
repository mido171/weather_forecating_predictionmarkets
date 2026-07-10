param(
  [string]$TaskName = "HKG-Tmax-Collector",
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
  throw "Python virtualenv not found: $Python"
}

$Action = New-ScheduledTaskAction -Execute $Python -Argument "-m hkg_tmax --root `"$RepoRoot`" acquisition run-due" -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 3650)
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "HKG Tmax weather data acquisition collector" -Force | Out-Null
Write-Output "Installed scheduled task $TaskName for $RepoRoot"

param(
  [string]$TaskName = "HKG-Tmax-Collector",
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
  [ValidateRange(30, 1440)][int]$IntervalMinutes = 30,
  [ValidateRange(1, 90)][int]$DurationDays = 30,
  [switch]$Execute,
  [switch]$Replace
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
  throw "Python virtualenv not found: $Python"
}

$Arguments = "-m hkg_tmax --root `"$RepoRoot`" acquisition run-due --execute"
if (-not $Execute) {
  Write-Output "DRY RUN: would register disabled task '$TaskName' every $IntervalMinutes minutes for $DurationDays days."
  Write-Output "Executable: $Python"
  Write-Output "Arguments: $Arguments"
  Write-Output "Re-run with -Execute after enabling and narrowing collector_schedules.yaml."
  return
}

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing -and -not $Replace) {
  throw "Scheduled task '$TaskName' already exists. Review it and pass -Replace explicitly."
}
if ($Existing) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) `
  -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
  -RepetitionDuration (New-TimeSpan -Days $DurationDays)
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 1 `
  -RestartInterval (New-TimeSpan -Minutes 10) -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
  -Description "Disabled-by-default HKG Tmax acquisition collector" | Out-Null
Disable-ScheduledTask -TaskName $TaskName | Out-Null
Write-Output "Registered disabled task '$TaskName'. Review its action, configure budgets, then enable it manually."

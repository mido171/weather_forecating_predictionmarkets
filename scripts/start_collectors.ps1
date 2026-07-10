param(
  [string]$TaskName = "HKG-Tmax-Collector"
)

$ErrorActionPreference = "Stop"
if (-not (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue)) {
  throw "Scheduled task $TaskName is not installed"
}
Start-ScheduledTask -TaskName $TaskName
Write-Output "Started scheduled task $TaskName"

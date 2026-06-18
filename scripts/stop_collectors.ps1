param(
  [string]$TaskName = "HKG-Tmax-Collector"
)

$ErrorActionPreference = "Stop"
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Write-Output "Stopped scheduled task $TaskName if it was running"

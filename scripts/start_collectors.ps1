param(
  [string]$TaskName = "HKG-Tmax-Collector"
)

$ErrorActionPreference = "Stop"
Start-ScheduledTask -TaskName $TaskName
Write-Output "Started scheduled task $TaskName"

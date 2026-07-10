param(
  [string]$TaskName = "HKG-Tmax-Collector"
)

$ErrorActionPreference = "Stop"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  Write-Output "Uninstalled scheduled task $TaskName"
} else {
  Write-Output "Scheduled task $TaskName is not installed"
}

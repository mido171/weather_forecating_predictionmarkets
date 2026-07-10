param(
  [string]$SqliteRoot,
  [switch]$NoWeb,
  [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
if ([string]::IsNullOrWhiteSpace($SqliteRoot)) {
  $SqliteRoot = Join-Path $repoRoot "var\ingestion\knyc"
}
$timestamp = Get-Date -Format "yyyyMMddTHHmmssZ"
$logDir = Join-Path $SqliteRoot "logs"
$logFile = Join-Path $logDir ("knyc-pilot-" + $timestamp + ".log")

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$argsList = @(
  "-pl", "apps/ingestion-service",
  "-DskipTests",
  "package",
  "dependency:build-classpath",
  "-Dmdep.outputFile=target\\pilot.classpath"
)

$runArgs = @(
  "--pilot.knyc.enabled=true",
  "--pilot.knyc.sqlite-root=$SqliteRoot",
  "--logging.level.org.springframework.boot=INFO",
  "--logging.level.org.springframework=INFO"
)
if ($NoWeb) {
  $runArgs += "--spring.main.web-application-type=none"
}
if ($ExtraArgs.Count -gt 0) {
  $runArgs += $ExtraArgs
}

Push-Location $repoRoot
try {
  "Writing pilot log to $logFile"
  & mvn @argsList 2>&1 | Tee-Object -FilePath $logFile
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }

  $classpathFile = Join-Path $scriptDir "target\\pilot.classpath"
  $classesDir = Join-Path $scriptDir "target\\classes"
  $dependencyCp = Get-Content -Path $classpathFile -Raw
  $classpath = "$classesDir;$dependencyCp"

  & java "-cp" $classpath "com.predictionmarkets.weather.pilot.PilotIngestionApplication" @runArgs 2>&1 | Tee-Object -FilePath $logFile -Append
} finally {
  Pop-Location
}

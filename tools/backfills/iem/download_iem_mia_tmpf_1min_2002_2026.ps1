# download_iem_mia_tmpf_1min_2002_2026.ps1
# Multi-station downloader for IEM/NCEI ASOS 1-minute air temperature (tmpf) in UTC.
#
# Default stations (Kalshi/NWS Daily Climate Report locations):
#   - MIA (Miami Intl Airport)
#   - DCA (Washington/National)
#   - NYC (New York City / Central Park)  IMPORTANT: IEM station code is NYC (NOT KNYC)
#   - LAX (Los Angeles Intl)
#   - PHL (Philadelphia Intl)
#
# For each station and each year 2002-2026 (inclusive), downloads one CSV per year:
#   data/iem_minute_data/<STATION>/tmpf/UTC/yearly/<STATION>_tmpf_1min_UTC_<YEAR>.csv
#
# Per-station manifest (auditable):
#   data/iem_minute_data/<STATION>/tmpf/UTC/meta/manifest_<STATION>_tmpf_1min_UTC_2002_2026.csv
#
# Per-station log:
#   data/iem_minute_data/<STATION>/tmpf/UTC/logs/download.log
#
# Source endpoint documentation:
#   https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?help=

[CmdletBinding()]
param(
  # List of IEM station identifiers. Accepts either:
  #   -Stations MIA DCA NYC LAX PHL
  # or:
  #   -Stations "MIA,DCA,NYC,LAX,PHL"
  [Parameter(Mandatory = $false)]
  [string[]]$Stations = @("MIA", "DCA", "NYC", "LAX", "PHL"),

  # Root directory where data is written. By default, this script writes under its own folder
  # (repo-relative: data/iem_minute_data/).
  [Parameter(Mandatory = $false)]
  [string]$OutputRoot = $PSScriptRoot,

  # If set, re-download ALL years even if the destination file already exists and looks valid.
  [Parameter(Mandatory = $false)]
  [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure TLS 1.2 on older Windows hosts
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {
  # If this fails, continue; many environments default to a safe protocol already.
}

# ----------------------------
# Fixed requirements
# ----------------------------
$Var       = "tmpf"  # Air Temperature [F] in IEM ASOS 1-minute interface
$Tz        = "UTC"   # Store in UTC for consistent minute counts
$YearStart = 2002
$YearEnd   = 2026

$Endpoint = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"

# Polite settings: retry + backoff + short delay between years
$MaxAttempts = 6
$BaseSleepSeconds = 3       # used for exponential backoff
$InterYearSleepSeconds = 2  # pause between yearly requests (per station)

# Optional: identify your script via User-Agent
$Headers = @{
  "User-Agent" = "weather-forecasting-predictionmarkets (IEM ASOS1min downloader)"
}

# ----------------------------
# Helpers
# ----------------------------
function Sanitize-PathToken([string]$s) {
  return ($s -replace "[\\/:*?`"<>| ]", "_")
}

function Ensure-Dir([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
  }
}

# Per-station log file is set at runtime by Process-Station
$script:LogFile = $null

function Log([string]$msg, [ValidateSet("INFO","WARN","ERROR")] [string]$level = "INFO") {
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "$ts  [$level]  $msg"
  Write-Host $line
  if ($script:LogFile) {
    Add-Content -LiteralPath $script:LogFile -Value $line -Encoding UTF8
  }
}

function Escape([string]$s) {
  return [uri]::EscapeDataString($s)
}

function Build-IemUrl([string]$station, [string]$vars, [string]$sts, [string]$ets, [string]$tz) {
  # IEM supports sts/ets ISO timestamps and tz applies to input/output timestamps.
  # Always request "download" and comma-delimited CSV.
  $qs = @(
    "station=$(Escape $station)"
    "vars=$(Escape $vars)"
    "sts=$(Escape $sts)"
    "ets=$(Escape $ets)"
    "what=download"
    "tz=$(Escape $tz)"
    "delim=comma"
  ) -join "&"
  return "${Endpoint}?$qs"
}

function Test-CsvLooksValid([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return $false }
  try {
    $len = (Get-Item -LiteralPath $path).Length
    if ($len -lt 20) { return $false }

    $firstLine = Get-Content -LiteralPath $path -TotalCount 1 -ErrorAction Stop
    if ([string]::IsNullOrWhiteSpace($firstLine)) { return $false }

    if ($firstLine -match "<html" -or $firstLine -match "<!DOCTYPE") { return $false }

    # Typical header begins with "station,valid(...", but we accept any CSV starting with "station,"
    if ($firstLine -notmatch "^station,") { return $false }

    return $true
  } catch {
    return $false
  }
}

function Replace-FileAtomically([string]$sourcePath, [string]$destPath) {
  # Ensures we never leave a corrupt/partial final file behind:
  # - If destination exists, use [System.IO.File]::Replace for atomic replacement (same volume).
  # - Otherwise, Move-Item is an atomic rename within a directory.
  if (Test-Path -LiteralPath $destPath) {
    $backup = "$destPath.bak"
    try {
      [System.IO.File]::Replace($sourcePath, $destPath, $backup, $true)
    } catch {
      # Fallback: remove + move (should be rare, but avoids leaving source behind)
      Remove-Item -LiteralPath $destPath -Force -ErrorAction SilentlyContinue
      Move-Item -LiteralPath $sourcePath -Destination $destPath -Force
    } finally {
      if (Test-Path -LiteralPath $backup) {
        Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
      }
    }
  } else {
    Move-Item -LiteralPath $sourcePath -Destination $destPath -Force
  }
}

function Download-WithRetry([string]$url, [string]$outFile) {
  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Log "GET (attempt $attempt/$MaxAttempts) -> $url"

      # Download to a temp file first to avoid corrupt partial final files
      $tmp = "$outFile.partial"
      if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }

      Invoke-WebRequest -Uri $url -Headers $Headers -OutFile $tmp -TimeoutSec 600

      if (-not (Test-CsvLooksValid $tmp)) {
        $firstLine = ""
        try { $firstLine = (Get-Content -LiteralPath $tmp -TotalCount 1 -ErrorAction SilentlyContinue) } catch {}
        throw "Downloaded file failed validation (not CSV or too small). First line: $firstLine"
      }

      # Warn (but don't fail) if header looks unexpected
      $firstLine = Get-Content -LiteralPath $tmp -TotalCount 1
      if ($firstLine -notmatch "^station,valid\(" -and $firstLine -notmatch "^station,valid,") {
        Log "Unexpected header (continuing): $firstLine" "WARN"
      }

      Replace-FileAtomically -sourcePath $tmp -destPath $outFile

      Log "OK: Saved -> $outFile"
      return
    }
    catch {
      $err = $_.Exception.Message
      Log "Download error: $err" "ERROR"
      if ($attempt -eq $MaxAttempts) {
        throw
      }
      $sleep = [math]::Min(120, ($BaseSleepSeconds * [math]::Pow(2, ($attempt - 1))))
      Log "Sleeping $sleep seconds before retry..."
      Start-Sleep -Seconds $sleep
    }
  }
}

function Get-Sha256([string]$path) {
  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ValidColumnIndex([string]$headerLine) {
  $cols = $headerLine.Split(",")
  for ($i = 0; $i -lt $cols.Length; $i++) {
    if ($cols[$i].StartsWith("valid(") -or $cols[$i] -eq "valid") {
      return $i
    }
  }
  # Default to column index 1 (historical assumption: station,valid(...),...)
  return 1
}

function Count-CsvDataRows([string]$path) {
  # Counts data rows excluding the first header line, using streaming for large files
  $count = 0
  $sr = New-Object System.IO.StreamReader($path)
  try {
    # Skip header
    $null = $sr.ReadLine()
    while (-not $sr.EndOfStream) {
      $null = $sr.ReadLine()
      $count++
    }
  } finally {
    $sr.Close()
  }
  return $count
}

function Get-FirstDataTimestamp([string]$path) {
  # Reads the first data line and returns the timestamp field (valid column)
  $sr = New-Object System.IO.StreamReader($path)
  try {
    $header = $sr.ReadLine()
    $idx = Get-ValidColumnIndex $header
    while (-not $sr.EndOfStream) {
      $line = $sr.ReadLine()
      if ([string]::IsNullOrWhiteSpace($line)) { continue }
      $parts = $line.Split(",")
      if ($parts.Length -gt $idx) { return $parts[$idx] }
    }
  } finally {
    $sr.Close()
  }
  return ""
}

function Get-LastDataTimestamp([string]$path) {
  # Uses Get-Content -Tail for efficiency; returns timestamp from last non-empty line
  $header = Get-Content -LiteralPath $path -TotalCount 1
  $idx = Get-ValidColumnIndex $header
  $tail = Get-Content -LiteralPath $path -Tail 50
  for ($i = $tail.Count - 1; $i -ge 0; $i--) {
    $line = $tail[$i]
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line -match "^station,") { continue } # skip header if file is tiny
    $parts = $line.Split(",")
    if ($parts.Length -gt $idx) { return $parts[$idx] }
  }
  return ""
}

function Try-ParseUtc([string]$s) {
  # Input often looks like "2025-08-01 00:00" when tz=UTC (no Z).
  # Parse and treat as UTC explicitly.
  try {
    $dt = [DateTime]::Parse($s, [System.Globalization.CultureInfo]::InvariantCulture)
    return [DateTime]::SpecifyKind($dt, [DateTimeKind]::Utc)
  } catch {
    return $null
  }
}

function Write-TextFileAtomically([string]$path, [string[]]$lines) {
  $tmp = "$path.partial"
  if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
  $lines | Set-Content -LiteralPath $tmp -Encoding UTF8

  Replace-FileAtomically -sourcePath $tmp -destPath $path
}

function Normalize-Stations([string[]]$stationsIn) {
  # Allow comma-separated single argument
  $tmp = @()
  foreach ($s in $stationsIn) {
    if ($null -eq $s) { continue }
    $text = $s.ToString().Trim()
    if ($text.Length -eq 0) { continue }
    if ($text.Contains(",")) {
      $tmp += $text.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries) | ForEach-Object { $_.Trim() }
    } else {
      $tmp += $text
    }
  }

  # Uppercase + de-dup while preserving order
  $seen = @{}
  $out = New-Object System.Collections.Generic.List[string]
  foreach ($s in $tmp) {
    $u = $s.ToUpperInvariant()
    if (-not $seen.ContainsKey($u)) {
      $seen[$u] = $true
      $out.Add($u) | Out-Null
    }
  }
  return ,$out.ToArray()
}

# ----------------------------
# Main per-station
# ----------------------------
function Process-Station([string]$Station, [string]$BaseDir) {
  $TzToken = Sanitize-PathToken $Tz

  $OutYearlyDir = Join-Path $BaseDir (Join-Path (Join-Path $Station (Join-Path $Var $TzToken)) "yearly")
  $OutMetaDir   = Join-Path $BaseDir (Join-Path (Join-Path $Station (Join-Path $Var $TzToken)) "meta")
  $OutLogDir    = Join-Path $BaseDir (Join-Path (Join-Path $Station (Join-Path $Var $TzToken)) "logs")

  Ensure-Dir $OutYearlyDir
  Ensure-Dir $OutMetaDir
  Ensure-Dir $OutLogDir

  $script:LogFile = Join-Path $OutLogDir "download.log"
  Log "============================================================"
  Log "Starting station=$Station vars=$Var tz=$Tz years=$YearStart-$YearEnd force=$Force"

  $ManifestFile = Join-Path $OutMetaDir ("manifest_{0}_{1}_1min_{2}_{3}_{4}.csv" -f $Station,$Var,$TzToken,$YearStart,$YearEnd)

  $manifestLines = New-Object System.Collections.Generic.List[string]
  $manifestLines.Add("year,file_path,bytes,sha256,data_rows,first_ts,last_ts,status,notes") | Out-Null

  $nowUtc = [DateTime]::UtcNow

  for ($year = $YearStart; $year -le $YearEnd; $year++) {
    $sts = "{0}-01-01T00:00Z" -f $year
    $ets = "{0}-01-01T00:00Z" -f ($year + 1)

    $url = Build-IemUrl -station $Station -vars $Var -sts $sts -ets $ets -tz $Tz

    $fileName = ("{0}_{1}_1min_{2}_{3}.csv" -f $Station,$Var,$TzToken,$year)
    $outFile  = Join-Path $OutYearlyDir $fileName

    # file_path stored in manifest is relative to $BaseDir (portable)
    $relPath  = Join-Path $Station (Join-Path (Join-Path $Var $TzToken) (Join-Path "yearly" $fileName))

    $status = "OK"
    $notes = ""

    try {
      $yearEndUtc = ([DateTime]::ParseExact(
        $ets,
        "yyyy-MM-ddTHH:mmZ",
        [System.Globalization.CultureInfo]::InvariantCulture,
        [System.Globalization.DateTimeStyles]::AssumeUniversal
      )).ToUniversalTime()
      $expectedLastFull = $yearEndUtc.AddMinutes(-1)
      $yearIsInProgress = ($yearEndUtc -gt $nowUtc.AddMinutes(1))

      $shouldDownload = $true
      if (-not $Force) {
        if ($yearIsInProgress) {
          # In-progress year (e.g., 2026): re-download each run to keep it fresh.
          $shouldDownload = $true
        } else {
          # Completed years: skip if existing file looks valid.
          if (Test-CsvLooksValid $outFile) {
            $shouldDownload = $false
            Log "Year ${year}: file already exists and looks valid; skipping download."
          }
        }
      }

      if ($shouldDownload) {
        Log "Year ${year}: downloading..."
        Download-WithRetry -url $url -outFile $outFile
      }

      # Compute audit stats from the on-disk file (whether newly downloaded or reused)
      if (-not (Test-CsvLooksValid $outFile)) {
        throw "Final output file failed validation after download/skip (file missing, too small, or not CSV)."
      }

      $bytes   = (Get-Item -LiteralPath $outFile).Length
      $sha     = Get-Sha256 -path $outFile
      $rows    = Count-CsvDataRows -path $outFile
      $firstTs = Get-FirstDataTimestamp -path $outFile
      $lastTs  = Get-LastDataTimestamp  -path $outFile

      if ($rows -le 0 -or [string]::IsNullOrWhiteSpace($firstTs) -or [string]::IsNullOrWhiteSpace($lastTs)) {
        $status = "ERROR"
        $notes = "No data rows or missing first/last timestamp."
      } else {
        $lastDt = Try-ParseUtc $lastTs

        if ($lastDt -eq $null) {
          $status = "WARN"
          $notes = "Could not parse last_ts as UTC; check output format."
        } elseif ($yearIsInProgress) {
          $status = "PARTIAL"
          $notes = "Year not complete (ets is in the future); file will grow as the archive updates."
          if ($lastDt -lt $nowUtc.AddDays(-3)) {
            $notes += " last_ts is >3 days behind current UTC; possible archive delay/outage."
          }
        } elseif ($lastDt -lt $expectedLastFull.AddDays(-2)) {
          $status = "PARTIAL"
          $notes = "Data ends earlier than requested (missing data/outage or archive cutoff)."
        } else {
          $status = "OK"
          $notes = ""
        }
      }

      # Write manifest line (CSV-escape file path and notes)
      $fp = $relPath.Replace('"','""')
      $n  = $notes.Replace('"','""')
      $manifestLines.Add("$year,""$fp"",$bytes,$sha,$rows,""$firstTs"",""$lastTs"",$status,""$n""") | Out-Null

      Log "Year $year done: rows=$rows last=$lastTs status=$status"
    }
    catch {
      $status = "ERROR"
      $notes = $_.Exception.Message
      $fp = $relPath.Replace('"','""')
      $n  = $notes.Replace('"','""')
      Log "Year $year failed: $notes" "ERROR"

      # Still write a manifest entry
      $manifestLines.Add("$year,""$fp"",0,,0,,,$status,""$n""") | Out-Null
    }

    Start-Sleep -Seconds $InterYearSleepSeconds
  }

  # Atomically write/replace the manifest so it's never half-written
  Write-TextFileAtomically -path $ManifestFile -lines $manifestLines.ToArray()

  Log "Finished station=$Station"
  Log "Yearly files -> $OutYearlyDir"
  Log "Manifest -> $ManifestFile"
}

# ----------------------------
# Entrypoint
# ----------------------------
$Stations = Normalize-Stations $Stations

# Resolve/create output root (portable; default is $PSScriptRoot)
try {
  $BaseDir = (Resolve-Path -LiteralPath $OutputRoot).Path
} catch {
  Ensure-Dir $OutputRoot
  $BaseDir = (Resolve-Path -LiteralPath $OutputRoot).Path
}

Write-Host ("OutputRoot: {0}" -f $BaseDir)
Write-Host ("Stations: {0}" -f ($Stations -join ","))

$stationErrors = 0
foreach ($st in $Stations) {
  try {
    Process-Station -Station $st -BaseDir $BaseDir
  } catch {
    $stationErrors++
    $msg = $_.Exception.Message
    Write-Host ("ERROR: Station {0} failed unexpectedly: {1}" -f $st, $msg)
  }
}

if ($stationErrors -gt 0) {
  Write-Host ("Completed with {0} station-level error(s)." -f $stationErrors)
  exit 1
}

Write-Host "All stations completed."
exit 0

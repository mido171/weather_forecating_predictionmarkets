$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-NativeOutputChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,

        [string[]] $Arguments = @()
    )

    $Output = & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
    return (($Output -join "`n").Trim())
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required to verify the standalone repository root."
}
$RepoRoot = Get-NativeOutputChecked -FilePath "git" -Arguments @(
    "-c", "core.fsmonitor=false", "rev-parse", "--show-toplevel"
)
$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
$ExpectedProjectRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot "projects\hkg-tmax"))
$ActualProjectRoot = [IO.Path]::GetFullPath($Root)
$GitDirectory = Join-Path $RepoRoot ".git"
$GitDirectoryItem = if (Test-Path -LiteralPath $GitDirectory -PathType Container) {
    Get-Item -LiteralPath $GitDirectory -Force
} else {
    $null
}
if (
    (Split-Path -Leaf $RepoRoot) -ne "weather_data_extraction" -or
    -not [string]::Equals($ActualProjectRoot, $ExpectedProjectRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $null -eq $GitDirectoryItem -or
    ($GitDirectoryItem.Attributes -band [IO.FileAttributes]::ReparsePoint)
) {
    throw "Refusing bootstrap outside the canonical weather_data_extraction root."
}
$FsMonitor = Get-NativeOutputChecked -FilePath "git" -Arguments @(
    "-C", $RepoRoot, "config", "--local", "--get", "core.fsmonitor"
)
if ($FsMonitor -ne "false") {
    throw "Local core.fsmonitor must be false before bootstrap."
}

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

Invoke-NativeChecked $Python -c "import sys; assert sys.version_info >= (3,11), sys.version"

if (-not (Test-Path ".venv")) {
    Invoke-NativeChecked $Python -m venv .venv
}

Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m pip install -e ".[research,dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add contact/API values where needed."
}

Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m hkg_tmax doctor
Invoke-NativeChecked ".\.venv\Scripts\python.exe" ..\..\tools\repo\doctor.py `
    --root ..\.. --scope projects/hkg-tmax
Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m pytest -q `
    tests/test_bootstrap_safety_contract.py `
    tests/test_config_and_sources.py `
    tests/test_experiments.py `
    tests/test_validation.py `
    tests/test_hko_backfill.py `
    tests/hkg_t24/test_h24n_contract_policy.py `
    tests/hkg_t24/test_schema_sql_contract.py `
    tests/test_demo_trading_migration.py
Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m hkg_tmax validate all
Invoke-NativeChecked ".\.venv\Scripts\python.exe" scripts/manage_campaign_documentation.py check

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next: follow AGENTS.md section 2, then read START_HERE.md and README.md."

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

Invoke-NativeChecked $Python -c "import sys; assert sys.version_info >= (3,11), sys.version"

if (-not (Test-Path ".venv")) {
    Invoke-NativeChecked $Python -m venv .venv
}

Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m pip install -e ".[research,dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add contact/API values where needed."
}

if (-not (Test-Path ".git")) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Invoke-NativeChecked "git" init
    }
}

Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m hkg_tmax doctor
Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m pytest
Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m hkg_tmax validate all
Invoke-NativeChecked ".\.venv\Scripts\python.exe" -m hkg_tmax manifest

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next: read CODEX_START_HERE.md and execute FIRST_GOALS.md."

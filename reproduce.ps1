param(
    [switch]$Full,
    [switch]$Wolfram
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$watch = [Diagnostics.Stopwatch]::StartNew()

function Invoke-ReproductionStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host "[RUN] $Name"
    & python -m kasner_scattering @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Host "[OK]  $Name"
}

Push-Location $root
try {
    Invoke-ReproductionStep "committed evidence" @("verify")
    if ($Full) {
        Invoke-ReproductionStep "full reconstruction" @(
            "rebuild", "--output", "build/full"
        )
    }
    Invoke-ReproductionStep "summary figure" @(
        "plot", "--output", "build/figures"
    )
    if ($Wolfram) {
        Invoke-ReproductionStep "independent Wolfram calculation" @(
            "crosscheck-wolfram"
        )
    }
} finally {
    Pop-Location
}
$watch.Stop()
Write-Host "Completed in $([math]::Round($watch.Elapsed.TotalSeconds, 1)) s"


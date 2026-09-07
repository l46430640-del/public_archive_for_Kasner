param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
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
            "rebuild", "--study", "all", "--output", "build/full"
        )
    }
    Invoke-ReproductionStep "response figure" @(
        "plot", "--output", "build/figures"
    )
} finally {
    Pop-Location
}
$watch.Stop()
Write-Host "Completed in $([math]::Round($watch.Elapsed.TotalSeconds, 1)) s"

<#
.SYNOPSIS
    Full paper reproduction pipeline across all HGB datasets.

.DESCRIPTION
    For each dataset:
      1. Mines metapaths via AnyBURL (10s snapshot — paper setting)
         Skipped if cached rules exist; use -ForceRemine to override.
      2. Runs all paper experiments (Table III, Table IV, Figures 4/5/6)
         across up to -MaxMetapaths validated metapaths.
      3. Saves results to results/<DATASET>/*.csv

    After all datasets complete, generates figures and table summaries
    automatically via scripts/plot_results.py.

    A master transcript is saved to results/run_<timestamp>.txt so you
    can read the full log when you come back.

    Results are resume-safe — interrupted runs pick up where they left off.

.PARAMETER SkipSweeps
    Pass -SkipSweeps to skip Figure 5 & 6 lambda/k sweeps.
    Much faster; only Table III, Table IV, and Figure 4 are produced.

.PARAMETER MaxMetapaths
    Maximum metapaths to process per dataset (default: 100).

.PARAMETER ForceRemine
    Re-run AnyBURL even if cached rules already exist.

.PARAMETER BoolapBinary
    Path to the compiled BoolAPCoreD binary.
    If provided, adds a BoolAP row to Table IV for each metapath.
    Path to the binary WITHOUT the "wsl" prefix — the script adds that automatically.
    Example: ".\parallel-k-P-core-decomposition-code\BoolAPCoreD"

.PARAMETER BoolapPlusBinary
    Path to the compiled BoolAPCoreG binary (BoolAP+ variant).
    If provided, adds a BoolAP+ row to Table IV for each metapath.

.EXAMPLE
    # Full run (all experiments, all datasets, 100 metapaths each)
    .\run_exhaustive.ps1

    # Quick run — no sweeps
    .\run_exhaustive.ps1 -SkipSweeps

    # Re-mine with paper config (10s) and run
    .\run_exhaustive.ps1 -ForceRemine

    # Custom cap
    .\run_exhaustive.ps1 -MaxMetapaths 50
#>

param(
    [switch]$SkipSweeps,
    [int]$MaxMetapaths = 10,
    [switch]$ForceRemine,
    [int]$Timeout          = 600,
    [string]$BoolapBinary     = "",
    [string]$BoolapPlusBinary = ""
)


if (-not (Test-Path "scripts/run_paper_experiments.py")) {
    Write-Error "Must be run from the project root directory."
    exit 1
}


$null = New-Item -ItemType Directory -Force -Path "results"
$Timestamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = "results/run_$Timestamp.txt"
Start-Transcript -Path $TranscriptPath -NoClobber


$Datasets = @("HGB_ACM", "HGB_DBLP", "HGB_IMDB", "HGB_Freebase")


Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   PAPER REPRODUCTION PIPELINE" -ForegroundColor Cyan
Write-Host "   Started : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
if ($SkipSweeps) {
    Write-Host "   Mode    : Table III + Table IV + Figure 4 only (--skip-sweeps)" -ForegroundColor Yellow
} else {
    Write-Host "   Mode    : Full (Table III + Table IV + Figures 4, 5, 6)" -ForegroundColor Yellow
}
Write-Host "   Max metapaths per dataset : $MaxMetapaths" -ForegroundColor Yellow
Write-Host "   Mining timeout            : 10s (paper setting)" -ForegroundColor Yellow
Write-Host "   C++ per-call timeout     : $($Timeout)s" -ForegroundColor Yellow
Write-Host "   Log file                  : $TranscriptPath" -ForegroundColor Yellow
Write-Host "============================================================`n" -ForegroundColor Cyan


$Failed = @()

foreach ($ds in $Datasets) {
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[Dataset] $ds  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow

    $pyargs = "scripts/run_paper_experiments.py $ds --max-metapaths $MaxMetapaths --mining-timeout 10 --timeout $Timeout"
    if ($SkipSweeps)           { $pyargs += " --skip-sweeps" }
    if ($ForceRemine)          { $pyargs += " --force-remine" }
    if ($BoolapBinary)         { $pyargs += " --boolap-binary `"$BoolapBinary`"" }
    if ($BoolapPlusBinary)     { $pyargs += " --boolap-plus-binary `"$BoolapPlusBinary`"" }

    Write-Host " -> python $pyargs" -ForegroundColor Gray
    Invoke-Expression "python $pyargs"

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "FAILURE on $ds (exit $LASTEXITCODE). Continuing with next dataset..."
        $Failed += $ds
    } else {
        Write-Host " [Done] $ds  ->  results/$ds/" -ForegroundColor Green
    }
    Write-Host ""
}


Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   GENERATING FIGURES + TABLE SUMMARIES" -ForegroundColor Cyan
Write-Host "   $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

python scripts/plot_results.py

if ($LASTEXITCODE -ne 0) {
    Write-Warning "plot_results.py failed (exit $LASTEXITCODE). Check that matplotlib and pandas are installed."
} else {
    Write-Host "`n [Done] Figures saved to results/figures/" -ForegroundColor Green
}


Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   ALL DONE  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "   Results   : results/" -ForegroundColor Cyan
Write-Host "   Figures   : results/figures/" -ForegroundColor Cyan
Write-Host "   Full log  : $TranscriptPath" -ForegroundColor Cyan
if ($Failed.Count -gt 0) {
    Write-Host "   FAILED    : $($Failed -join ', ')" -ForegroundColor Red
} else {
    Write-Host "   All datasets completed successfully." -ForegroundColor Green
}
Write-Host "============================================================`n" -ForegroundColor Cyan

Stop-Transcript

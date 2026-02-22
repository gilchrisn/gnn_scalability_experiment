<#
.SYNOPSIS
    Runs the GNN Scalability Master Protocol exhaustively across datasets and models.
#>

# --- Configuration ---
$Datasets = @("HGB_IMDB", "HGB_IMDB", "HGB_Freebase", "HGB_ACM")
$Models   = @("SAGE", "GCN")

# New Multi-Path Hyperparameters
$MiningStrategy = "stratified"
$StratifiedBuckets = "2 4 6 8"
$MinConf = 0.001
$TopPaths = 2

# Ensure correct working directory
if (-not (Test-Path "scripts/run_master_protocol.py")) {
    Write-Error "Error: Script must be run from the project root directory."
    exit 1
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   STARTING EXHAUSTIVE EXPERIMENT PROTOCOL (CPU MODE)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

foreach ($ds in $Datasets) {
    foreach ($mod in $Models) {
        Write-Host "`n------------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "[Experiment] Dataset: $ds | Model: $mod" -ForegroundColor Yellow
        Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
        
        # Execute Protocol
        $cmd = "python scripts/run_master_protocol.py --dataset $ds --model $mod --cpu --epochs 50 --mining-strategy $MiningStrategy --buckets $StratifiedBuckets --min-conf $MinConf --top-paths $TopPaths"
        
        Write-Host " -> Launching Command: $cmd" -ForegroundColor Gray
        Invoke-Expression $cmd

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "FAILURE/OOM: Python script exited with code $LASTEXITCODE for $ds / $mod. Continuing to next configuration..."
            continue
        } else {
            Write-Host " [Success] Run completed for $ds / $mod" -ForegroundColor Green
        }
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "   GENERATING AGGREGATED PLOTS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (Test-Path "scripts/generate_journal_plots.py") {
    python scripts/generate_journal_plots.py
    Write-Host "`n[Success] Plots generated in output/plots/" -ForegroundColor Green
} else {
    Write-Warning "Plotting script not found."
}
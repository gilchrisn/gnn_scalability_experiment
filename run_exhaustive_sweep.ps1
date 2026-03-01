# run_exhaustive_sweep.ps1
#
# Exhaustive E2E sweep across all 4 HGB datasets x 2 metapaths each.
# ALL computation forced to CPU. Quality logs written per job.
#
# Usage:
#   .\run_exhaustive_sweep.ps1                            # full sweep
#   .\run_exhaustive_sweep.ps1 -DryRun                    # print commands only
#   .\run_exhaustive_sweep.ps1 -Datasets HGB_ACM,HGB_DBLP # subset
#   .\run_exhaustive_sweep.ps1 -SkipCompleted             # resume after crash
#   .\run_exhaustive_sweep.ps1 -Epochs 200 -L 10          # longer runs

param(
    [int]      $Epochs       = 100,
    [int[]]    $KValues      = @(2, 4, 8, 16, 32, 64, 128),
    [int]      $L            = 5,
    [string]   $Model        = "SAGE",
    [string]   $OutputCSV    = "output\results\exhaustive_sweep.csv",
    [string]   $LogDir       = "output\logs",
    [string]   $Python       = "python",
    [string]   $Script       = "scripts\run_e2e_comparison.py",
    [string[]] $Datasets     = @(),
    [switch]   $DryRun,
    [switch]   $SkipCompleted
)

$DatasetMetapaths = [ordered]@{
    'HGB_DBLP'     = @(
        "author_to_paper,paper_to_author",
        "author_to_paper,paper_to_venue,venue_to_paper,paper_to_author"
    )
    'HGB_ACM'      = @(
        "paper_to_author,author_to_paper",
        "paper_to_subject,subject_to_paper"
    )
    'HGB_IMDB'     = @(
        "movie_to_actor,actor_to_movie",
        "movie_to_director,director_to_movie"
    )
    'HGB_Freebase' = @(
        "book_to_people,people_to_book",
        "book_to_film,film_to_book"
    )
}

function Write-Header([string]$Text) {
    $line = "=" * 62
    Write-Host ""; Write-Host $line -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan; Write-Host $line -ForegroundColor Cyan
}

function Write-Job([string]$Text) {
    Write-Host ""; Write-Host "  >> $Text" -ForegroundColor Yellow
}

function Get-ShortName([string]$Metapath) {
    $parts = $Metapath -split ","; $letters = @()
    foreach ($p in $parts) {
        $t = $p.Trim() -split "_to_"
        if ($t.Count -ge 1) { $letters += $t[0][0].ToString().ToUpper() }
    }
    $last = ($parts[-1].Trim() -split "_to_")
    if ($last.Count -ge 2) { $letters += $last[-1][0].ToString().ToUpper() }
    return ($letters -join "")
}

function Get-CompletedKeys {
    $done = [System.Collections.Generic.HashSet[string]]::new()
    if (Test-Path $OutputCSV) {
        try { Import-Csv $OutputCSV | ForEach-Object { [void]$done.Add("$($_.dataset)|$($_.metapath)") } }
        catch { Write-Warning "Could not parse CSV: $_" }
    }
    return $done
}

Write-Header "EXHAUSTIVE HGB SWEEP  (CPU-forced)"
Write-Host "  Datasets : HGB_DBLP, HGB_ACM, HGB_IMDB, HGB_Freebase"
Write-Host "  Epochs   : $Epochs"
Write-Host "  K values : $($KValues -join ', ')"
Write-Host "  L        : $L"
Write-Host "  Model    : $Model"
Write-Host "  Device   : CPU (forced inside run_e2e_comparison.py)"
Write-Host "  Output   : $OutputCSV"
Write-Host "  Logs     : $LogDir\"
Write-Host "  DryRun   : $DryRun  |  SkipDone: $SkipCompleted"

try { $v = & $Python --version 2>&1; Write-Host "  Python   : $v" -ForegroundColor Green }
catch { Write-Error "Python not found at '$Python'"; exit 1 }

if (-not (Test-Path $Script)) { Write-Error "Script not found: $Script"; exit 1 }

New-Item -ItemType Directory -Force -Path (Split-Path $OutputCSV -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ActiveDatasets = if ($Datasets.Count -gt 0) {
    $DatasetMetapaths.Keys | Where-Object { $Datasets -contains $_ }
} else { $DatasetMetapaths.Keys }

if (-not $ActiveDatasets) { Write-Error "No matching datasets"; exit 1 }

$Jobs = [System.Collections.Generic.List[hashtable]]::new()
foreach ($ds in $ActiveDatasets) {
    foreach ($mp in $DatasetMetapaths[$ds]) {
        $Jobs.Add(@{ Dataset = $ds; Metapath = $mp; Short = Get-ShortName $mp })
    }
}

$runsPerJob = $KValues.Count + 1
Write-Host "  Jobs     : $($Jobs.Count)  ($($Jobs.Count * $runsPerJob) total pipeline runs)"

$completedKeys = if ($SkipCompleted) { Get-CompletedKeys } else { $null }
$jobIdx = 0; $ok = 0; $skip = 0; $fail = 0
$sweepStart = Get-Date

foreach ($job in $Jobs) {
    $jobIdx++
    $ds = $job.Dataset; $mp = $job.Metapath; $short = $job.Short
    $logFile = Join-Path $LogDir "${ds}_${short}.log"
    $kStr = $KValues -join " "

    Write-Job "[$jobIdx/$($Jobs.Count)]  $ds  |  $short"
    Write-Host "    Metapath: $mp"

    if ($SkipCompleted -and $completedKeys -and $completedKeys.Contains("${ds}|${mp}")) {
        Write-Host "    [SKIP] Already complete" -ForegroundColor DarkGray
        $skip++; continue
    }

    # CPU is forced inside the Python script — no --cpu flag needed
    $cmdArgs = @(
        $Script,
        "--dataset",    $ds,
        "--metapath",   $mp,
        "--model",      $Model,
        "--epochs",     $Epochs,
        "--k-values",   $kStr,
        "--l",          $L,
        "--output-csv", $OutputCSV
    )
    $cmdStr = "$Python $($cmdArgs -join ' ')"
    Write-Host "    CMD: $cmdStr" -ForegroundColor DarkGray
    Write-Host "    LOG: $logFile"

    if ($DryRun) { Write-Host "    [DRY RUN]" -ForegroundColor Magenta; continue }

    $jobStart = Get-Date
    try {
        $proc = Start-Process -FilePath $Python -ArgumentList $cmdArgs `
            -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput "${logFile}.out" `
            -RedirectStandardError  "${logFile}.err"

        $header = "=" * 60 + "`nDataset  : $ds`nMetapath : $mp ($short)`nStarted  : $jobStart`nCommand  : $cmdStr`n" + "=" * 60
        Set-Content $logFile $header
        if (Test-Path "${logFile}.out") { Get-Content "${logFile}.out" | Add-Content $logFile; Remove-Item "${logFile}.out" -Force }
        if (Test-Path "${logFile}.err") {
            $err = Get-Content "${logFile}.err" -Raw
            if ($err.Trim()) { Add-Content $logFile "`n--- STDERR ---`n$err" }
            Remove-Item "${logFile}.err" -Force
        }

        $elapsed = (Get-Date) - $jobStart
        if ($proc.ExitCode -eq 0) {
            Write-Host "    [OK] $($elapsed.ToString('mm\:ss'))" -ForegroundColor Green; $ok++
        } else {
            Write-Host "    [FAIL] exit=$($proc.ExitCode) $($elapsed.ToString('mm\:ss'))" -ForegroundColor Red
            Write-Host "    See: $logFile" -ForegroundColor Red; $fail++
        }
    } catch { Write-Host "    [ERROR] $_" -ForegroundColor Red; $fail++ }

    $done = $ok + $fail
    if ($done -gt 0) {
        $avg = ((Get-Date) - $sweepStart).TotalSeconds / $done
        $eta = [TimeSpan]::FromSeconds(($Jobs.Count - $jobIdx) * $avg)
        Write-Host "    ETA ~$($eta.ToString('hh\:mm\:ss')) for $($Jobs.Count - $jobIdx) remaining" -ForegroundColor DarkCyan
    }
}

$totalTime = (Get-Date) - $sweepStart
Write-Header "SWEEP COMPLETE"
Write-Host "  Wall time  : $($totalTime.ToString('hh\:mm\:ss'))"
Write-Host "  Succeeded  : $ok"   -ForegroundColor $(if ($ok   -gt 0) {'Green'} else {'White'})
Write-Host "  Failed     : $fail" -ForegroundColor $(if ($fail -gt 0) {'Red'  } else {'Green'})
Write-Host "  Skipped    : $skip" -ForegroundColor DarkGray
Write-Host "  CSV        : $OutputCSV"
Write-Host "  Logs       : $LogDir\"

if (Test-Path $OutputCSV) {
    try {
        $rows = Import-Csv $OutputCSV
        Write-Host "  CSV rows   : $($rows.Count)"
        $exactRows = $rows | Where-Object { $_.method -eq 'Exact' }
        $kmvRows   = $rows | Where-Object { $_.method -eq 'KMV'   }
        if ($exactRows) {
            $avg = ($exactRows | Measure-Object -Property test_acc -Average).Average
            Write-Host "  Avg Exact acc : $([math]::Round([double]$avg, 4))" -ForegroundColor Cyan
        }
        if ($kmvRows) {
            $avg = ($kmvRows | Measure-Object -Property test_acc -Average).Average
            Write-Host "  Avg KMV   acc : $([math]::Round([double]$avg, 4))" -ForegroundColor Cyan
            Write-Host "  Per-K accuracy:" -ForegroundColor White
            $kmvRows | Group-Object k | Sort-Object { [int]$_.Name } | ForEach-Object {
                $kAcc = ($_.Group | Measure-Object -Property test_acc -Average).Average
                Write-Host ("    K={0,-5}  avg_acc={1:F4}" -f $_.Name, [double]$kAcc) -ForegroundColor Gray
            }
        }
    } catch { Write-Warning "Could not parse CSV for summary: $_" }
}

if ($fail -gt 0) {
    Write-Host ""
    Write-Host "  Re-run failed jobs: .\run_exhaustive_sweep.ps1 -SkipCompleted" -ForegroundColor Yellow
}
Write-Host ""
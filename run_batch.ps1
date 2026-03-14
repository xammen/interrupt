<#
.SYNOPSIS
    Batch runner for the Interrupt Effect study.
    Runs all conditions sequentially with proper output capture.
#>

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResultsDir = Join-Path $ProjectDir "results"

$Prompt = @"
Write a Python module called task_scheduler.py that implements an async task scheduler with the following features:

1. A Task dataclass with fields: id (str), name (str), priority (int 1-10), dependencies (list of task IDs), status (enum: PENDING, RUNNING, COMPLETED, FAILED), retry_count (int), max_retries (int, default 3), created_at (datetime), result (Optional[Any])

2. A TaskScheduler class that:
   - Maintains a priority queue of tasks
   - Resolves dependency graphs before execution (topological sort)
   - Detects circular dependencies and raises CircularDependencyError
   - Runs independent tasks concurrently using asyncio
   - Implements exponential backoff retry logic for failed tasks
   - Has a configurable concurrency limit (max parallel tasks)
   - Emits events via an observer pattern (on_task_start, on_task_complete, on_task_fail)
   - Provides a get_execution_plan() method that returns the ordered execution groups
   - Tracks execution metrics (total time, per-task time, retry count)

3. Include proper error handling, type hints, and docstrings.

4. Write at least 5 unit tests using pytest that cover:
   - Basic task execution
   - Dependency resolution
   - Circular dependency detection
   - Retry logic with exponential backoff
   - Concurrent execution respecting concurrency limits

Write the complete code in a single response. Do not ask questions, just write the code.
"@

$FragmentContinueTemplate = @"
I was generating code but got interrupted. Here is what I had so far. Please continue EXACTLY from where this stops - do not repeat what's already written, just complete the remaining code. Keep the same style, variable names, and architecture.

--- PARTIAL CODE ---
PLACEHOLDER_FRAGMENT
--- END PARTIAL CODE ---

Continue the code from where it stopped. Write only the remaining part.
"@

$Models = @(
    @{ Name = "sonnet"; Id = "anthropic/claude-sonnet-4-6" },
    @{ Name = "opus"; Id = "anthropic/claude-opus-4-6" }
)

$NumRuns = 5
$InterruptTimeout = 20  # seconds

function Write-Status($msg, $color = "White") {
    Write-Host "[$([datetime]::Now.ToString('HH:mm:ss'))] $msg" -ForegroundColor $color
}

function Run-SingleGeneration {
    param([string]$Model, [string]$PromptText, [int]$TimeoutSec = 300)
    
    $tempFile = [System.IO.Path]::GetTempFileName()
    
    try {
        $process = Start-Process -FilePath "opencode" `
            -ArgumentList "run", "--model", $Model, $PromptText `
            -NoNewWindow -RedirectStandardOutput $tempFile -RedirectStandardError ([System.IO.Path]::GetTempFileName()) `
            -PassThru
        
        $exited = $process.WaitForExit($TimeoutSec * 1000)
        
        if (-not $exited) {
            $process.Kill()
            $output = Get-Content $tempFile -Raw -ErrorAction SilentlyContinue
            return @{ Output = $output; TimedOut = $true; Duration = $TimeoutSec }
        }
        
        $output = Get-Content $tempFile -Raw -ErrorAction SilentlyContinue
        return @{ Output = $output; TimedOut = $false; Duration = 0 }
    } finally {
        Remove-Item $tempFile -ErrorAction SilentlyContinue
    }
}

# ============================================
# MAIN
# ============================================

Write-Status "THE INTERRUPT EFFECT - Starting batch experiment" "Cyan"
Write-Status "Models: $($Models.Name -join ', ')" "Gray"
Write-Status "Runs per condition: $NumRuns" "Gray"
Write-Status "Interrupt timeout: ${InterruptTimeout}s" "Gray"
Write-Status ""

foreach ($model in $Models) {
    $modelName = $model.Name
    $modelId = $model.Id
    
    Write-Status "=== MODEL: $modelName ($modelId) ===" "Yellow"
    
    # --- CONDITION 1: COMPLETE ---
    Write-Status "--- Condition: COMPLETE ---" "Green"
    $outDir = Join-Path $ResultsDir "${modelName}_complete"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    
    for ($i = 1; $i -le $NumRuns; $i++) {
        Write-Status "  Run $i/$NumRuns (complete)..." "Green"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        
        $result = opencode run --model $modelId $Prompt 2>&1 | Out-String
        
        $sw.Stop()
        $result | Out-File (Join-Path $outDir "run_${i}.txt") -Encoding utf8
        
        @{
            model = $modelId; condition = "complete"; run_index = $i
            duration_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            timestamp = (Get-Date -Format "o")
            output_length = $result.Length
        } | ConvertTo-Json | Out-File (Join-Path $outDir "run_${i}_meta.json") -Encoding utf8
        
        Write-Status "    Done: $($result.Length) chars in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" "DarkGreen"
        Start-Sleep -Seconds 2
    }
    
    # --- CONDITION 2: INTERRUPTED (kill + re-generate from scratch) ---
    Write-Status "--- Condition: INTERRUPTED ---" "Yellow"
    $outDir = Join-Path $ResultsDir "${modelName}_interrupted"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    
    for ($i = 1; $i -le $NumRuns; $i++) {
        Write-Status "  Run $i/$NumRuns (interrupted)..." "Yellow"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        
        # Phase 1: Start and kill after timeout
        Write-Status "    Phase 1: Generate + kill after ${InterruptTimeout}s..." "DarkYellow"
        $tempOut = [System.IO.Path]::GetTempFileName()
        $tempErr = [System.IO.Path]::GetTempFileName()
        
        $proc = Start-Process -FilePath "opencode" `
            -ArgumentList @("run", "--model", $modelId, $Prompt) `
            -NoNewWindow -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr `
            -PassThru
        
        $exited = $proc.WaitForExit($InterruptTimeout * 1000)
        if (-not $exited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Status "    Phase 1: Killed after ${InterruptTimeout}s" "DarkYellow"
        } else {
            Write-Status "    Phase 1: Finished before timeout" "DarkYellow"
        }
        
        $partialOutput = Get-Content $tempOut -Raw -ErrorAction SilentlyContinue
        Remove-Item $tempOut, $tempErr -ErrorAction SilentlyContinue
        
        # Phase 2: Re-generate from scratch (fresh prompt, no fragment)
        Write-Status "    Phase 2: Fresh re-generation..." "DarkYellow"
        $result = opencode run --model $modelId $Prompt 2>&1 | Out-String
        
        $sw.Stop()
        $result | Out-File (Join-Path $outDir "run_${i}.txt") -Encoding utf8
        
        @{
            model = $modelId; condition = "interrupted"; run_index = $i
            duration_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            interrupt_timeout = $InterruptTimeout
            timestamp = (Get-Date -Format "o")
            output_length = $result.Length
            partial_output_length = if ($partialOutput) { $partialOutput.Length } else { 0 }
        } | ConvertTo-Json | Out-File (Join-Path $outDir "run_${i}_meta.json") -Encoding utf8
        
        Write-Status "    Done: $($result.Length) chars in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" "DarkYellow"
        Start-Sleep -Seconds 2
    }
    
    # --- CONDITION 3: FRAGMENT + CONTINUE ---
    Write-Status "--- Condition: FRAGMENT_CONTINUE ---" "Cyan"
    $outDir = Join-Path $ResultsDir "${modelName}_fragment_continue"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    
    for ($i = 1; $i -le $NumRuns; $i++) {
        Write-Status "  Run $i/$NumRuns (fragment+continue)..." "Cyan"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        
        # Phase 1: Start and kill after timeout to get fragment
        Write-Status "    Phase 1: Generate + kill after ${InterruptTimeout}s to get fragment..." "DarkCyan"
        $tempOut = [System.IO.Path]::GetTempFileName()
        $tempErr = [System.IO.Path]::GetTempFileName()
        
        $proc = Start-Process -FilePath "opencode" `
            -ArgumentList @("run", "--model", $modelId, $Prompt) `
            -NoNewWindow -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr `
            -PassThru
        
        $exited = $proc.WaitForExit($InterruptTimeout * 1000)
        if (-not $exited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Status "    Phase 1: Killed - got fragment" "DarkCyan"
        } else {
            Write-Status "    Phase 1: Finished before timeout (using as fragment anyway)" "DarkCyan"
        }
        
        $fragment = Get-Content $tempOut -Raw -ErrorAction SilentlyContinue
        Remove-Item $tempOut, $tempErr -ErrorAction SilentlyContinue
        
        # Save fragment
        $fragment | Out-File (Join-Path $outDir "run_${i}_fragment.txt") -Encoding utf8
        
        # Phase 2: Feed fragment back and ask to continue
        Write-Status "    Phase 2: Feeding fragment ($($fragment.Length) chars) back..." "DarkCyan"
        $continuePrompt = $FragmentContinueTemplate -replace 'PLACEHOLDER_FRAGMENT', $fragment
        
        $continuation = opencode run --model $modelId $continuePrompt 2>&1 | Out-String
        
        # Combine
        $combined = $fragment + "`n`n# === CONTINUATION AFTER INTERRUPT ===`n`n" + $continuation
        $combined | Out-File (Join-Path $outDir "run_${i}.txt") -Encoding utf8
        
        $sw.Stop()
        
        @{
            model = $modelId; condition = "fragment_continue"; run_index = $i
            duration_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            interrupt_timeout = $InterruptTimeout
            timestamp = (Get-Date -Format "o")
            output_length = $combined.Length
            fragment_length = if ($fragment) { $fragment.Length } else { 0 }
            continuation_length = $continuation.Length
        } | ConvertTo-Json | Out-File (Join-Path $outDir "run_${i}_meta.json") -Encoding utf8
        
        Write-Status "    Done: fragment=$($fragment.Length) + continuation=$($continuation.Length) chars in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" "DarkCyan"
        Start-Sleep -Seconds 2
    }
}

Write-Status ""
Write-Status "=========================================" "Green"
Write-Status "ALL EXPERIMENTS COMPLETE" "Green"
Write-Status "Results in: $ResultsDir" "Green"
Write-Status "Run analysis: python analyze_results.py" "Green"
Write-Status "=========================================" "Green"

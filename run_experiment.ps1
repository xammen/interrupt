<#
.SYNOPSIS
    The Interrupt Effect - Experiment Runner
    Tests how interrupting LLM code generation affects output quality.

.DESCRIPTION
    Runs opencode CLI with Anthropic models (Opus 4.6 & Sonnet 4.6).
    
    Conditions:
    A) COMPLETE    - Let the model generate fully, no interruption
    B) INTERRUPTED - Kill the generation mid-way, re-prompt from scratch  
    C) FRAGMENT    - Kill mid-way, feed partial output back + "continue from where you stopped"
    
    Each condition is run N times per model to measure variance.
#>

param(
    [int]$Runs = 5,
    [switch]$SkipOpus,
    [switch]$SkipSonnet,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$ProjectDir = $PSScriptRoot
$ResultsDir = Join-Path $ProjectDir "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# The exact prompt - read from PROMPT.md, extract just the task
$PromptFile = Join-Path $ProjectDir "PROMPT.md"
$RawPrompt = Get-Content $PromptFile -Raw
# Skip the markdown header line
$Prompt = ($RawPrompt -split "`n" | Select-Object -Skip 2) -join "`n"
$Prompt = $Prompt.Trim()

# Fragment continue prompt
$FragmentContinuePrompt = @"
I was generating code but got interrupted. Here is what I had so far. Please continue EXACTLY from where this stops - do not repeat what's already written, just complete the remaining code. Keep the same style, variable names, and architecture.

--- PARTIAL CODE ---
{FRAGMENT}
--- END PARTIAL CODE ---

Continue the code from where it stopped. Write only the remaining part.
"@

$Models = @()
if (-not $SkipOpus) { $Models += "anthropic/claude-opus-4-6" }
if (-not $SkipSonnet) { $Models += "anthropic/claude-sonnet-4-6" }

$Conditions = @("complete", "interrupted", "fragment_continue")

function Get-ModelShortName($model) {
    if ($model -match "opus") { return "opus" }
    if ($model -match "sonnet") { return "sonnet" }
    return "unknown"
}

function Run-Complete {
    param($Model, $RunIndex, $OutputDir)
    
    $outFile = Join-Path $OutputDir "run_${RunIndex}.txt"
    $metaFile = Join-Path $OutputDir "run_${RunIndex}_meta.json"
    
    Write-Host "  [COMPLETE] Run $RunIndex - Full generation..." -ForegroundColor Green
    
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    
    if ($DryRun) {
        "DRY RUN - would run: opencode run --model $Model" | Out-File $outFile
    } else {
        $result = opencode run --model $Model $Prompt 2>&1
        $result | Out-File $outFile -Encoding utf8
    }
    
    $sw.Stop()
    
    $meta = @{
        model = $Model
        condition = "complete"
        run_index = $RunIndex
        duration_seconds = $sw.Elapsed.TotalSeconds
        timestamp = (Get-Date -Format "o")
        output_length = (Get-Content $outFile -Raw).Length
    } | ConvertTo-Json
    
    $meta | Out-File $metaFile -Encoding utf8
    Write-Host "    Done in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s - $((Get-Content $outFile -Raw).Length) chars" -ForegroundColor DarkGreen
}

function Run-Interrupted {
    param($Model, $RunIndex, $OutputDir, $TimeoutSeconds = 15)
    
    $outFile = Join-Path $OutputDir "run_${RunIndex}.txt"
    $metaFile = Join-Path $OutputDir "run_${RunIndex}_meta.json"
    
    Write-Host "  [INTERRUPTED] Run $RunIndex - Generate, kill at ${TimeoutSeconds}s, re-generate..." -ForegroundColor Yellow
    
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    
    if ($DryRun) {
        "DRY RUN - would run interrupted" | Out-File $outFile
    } else {
        # Phase 1: Start generation, kill it after timeout
        Write-Host "    Phase 1: Starting generation (will kill after ${TimeoutSeconds}s)..." -ForegroundColor DarkYellow
        
        $job = Start-Job -ScriptBlock {
            param($m, $p)
            opencode run --model $m $p 2>&1
        } -ArgumentList $Model, $Prompt
        
        # Wait for timeout then kill
        $completed = Wait-Job $job -Timeout $TimeoutSeconds
        if ($null -eq $completed) {
            Stop-Job $job
            Write-Host "    Phase 1: Killed after ${TimeoutSeconds}s" -ForegroundColor DarkYellow
        } else {
            Write-Host "    Phase 1: Completed before timeout (fast generation)" -ForegroundColor DarkYellow
        }
        $partialOutput = Receive-Job $job 2>$null
        Remove-Job $job -Force 2>$null
        
        # Phase 2: Fresh re-generation (same prompt, no context of interruption)
        Write-Host "    Phase 2: Fresh re-generation..." -ForegroundColor DarkYellow
        $result = opencode run --model $Model $Prompt 2>&1
        $result | Out-File $outFile -Encoding utf8
    }
    
    $sw.Stop()
    
    $meta = @{
        model = $Model
        condition = "interrupted"
        run_index = $RunIndex
        duration_seconds = $sw.Elapsed.TotalSeconds
        timeout_seconds = $TimeoutSeconds
        timestamp = (Get-Date -Format "o")
        output_length = (Get-Content $outFile -Raw).Length
        partial_output_length = if ($partialOutput) { ($partialOutput | Out-String).Length } else { 0 }
    } | ConvertTo-Json
    
    $meta | Out-File $metaFile -Encoding utf8
    Write-Host "    Done in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" -ForegroundColor DarkYellow
}

function Run-FragmentContinue {
    param($Model, $RunIndex, $OutputDir, $TimeoutSeconds = 15)
    
    $outFile = Join-Path $OutputDir "run_${RunIndex}.txt"
    $fragFile = Join-Path $OutputDir "run_${RunIndex}_fragment.txt"
    $metaFile = Join-Path $OutputDir "run_${RunIndex}_meta.json"
    
    Write-Host "  [FRAGMENT+CONTINUE] Run $RunIndex - Generate, kill at ${TimeoutSeconds}s, feed fragment back..." -ForegroundColor Cyan
    
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    
    if ($DryRun) {
        "DRY RUN - would run fragment continue" | Out-File $outFile
    } else {
        # Phase 1: Start generation, kill it after timeout
        Write-Host "    Phase 1: Starting generation (will kill after ${TimeoutSeconds}s)..." -ForegroundColor DarkCyan
        
        $job = Start-Job -ScriptBlock {
            param($m, $p)
            opencode run --model $m $p 2>&1
        } -ArgumentList $Model, $Prompt
        
        $completed = Wait-Job $job -Timeout $TimeoutSeconds
        if ($null -eq $completed) {
            Stop-Job $job
            Write-Host "    Phase 1: Killed after ${TimeoutSeconds}s" -ForegroundColor DarkCyan
        }
        $partialRaw = Receive-Job $job 2>$null
        Remove-Job $job -Force 2>$null
        
        $fragment = $partialRaw | Out-String
        $fragment | Out-File $fragFile -Encoding utf8
        
        # Phase 2: Feed fragment back and ask to continue
        Write-Host "    Phase 2: Feeding fragment ($($fragment.Length) chars) back to model..." -ForegroundColor DarkCyan
        $continuePrompt = $FragmentContinuePrompt -replace '\{FRAGMENT\}', $fragment
        
        $continuation = opencode run --model $Model $continuePrompt 2>&1
        
        # Combine fragment + continuation
        $combined = $fragment + "`n`n# --- CONTINUATION FROM INTERRUPT ---`n`n" + ($continuation | Out-String)
        $combined | Out-File $outFile -Encoding utf8
    }
    
    $sw.Stop()
    
    $meta = @{
        model = $Model
        condition = "fragment_continue"
        run_index = $RunIndex
        duration_seconds = $sw.Elapsed.TotalSeconds
        timeout_seconds = $TimeoutSeconds
        timestamp = (Get-Date -Format "o")
        output_length = (Get-Content $outFile -Raw).Length
        fragment_length = if ($fragment) { $fragment.Length } else { 0 }
    } | ConvertTo-Json
    
    $meta | Out-File $metaFile -Encoding utf8
    Write-Host "    Done in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" -ForegroundColor DarkCyan
}

# ============================================
# MAIN EXECUTION
# ============================================

Write-Host ""
Write-Host "=========================================" -ForegroundColor White
Write-Host "  THE INTERRUPT EFFECT - Experiment Runner" -ForegroundColor White
Write-Host "=========================================" -ForegroundColor White
Write-Host ""
Write-Host "Config:" -ForegroundColor Gray
Write-Host "  Models: $($Models -join ', ')" -ForegroundColor Gray
Write-Host "  Runs per condition: $Runs" -ForegroundColor Gray
Write-Host "  Conditions: $($Conditions -join ', ')" -ForegroundColor Gray
Write-Host "  Total runs: $($Models.Count * $Conditions.Count * $Runs)" -ForegroundColor Gray
Write-Host "  DryRun: $DryRun" -ForegroundColor Gray
Write-Host ""

$totalRuns = $Models.Count * $Conditions.Count * $Runs
$currentRun = 0

foreach ($model in $Models) {
    $shortName = Get-ModelShortName $model
    Write-Host "=== Model: $model ===" -ForegroundColor White
    
    foreach ($condition in $Conditions) {
        $outputDir = Join-Path $ResultsDir "${shortName}_${condition}"
        
        # Ensure directory exists
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        for ($i = 1; $i -le $Runs; $i++) {
            $currentRun++
            Write-Host ""
            Write-Host "[$currentRun/$totalRuns] $shortName / $condition / run $i" -ForegroundColor White
            
            switch ($condition) {
                "complete" {
                    Run-Complete -Model $model -RunIndex $i -OutputDir $outputDir
                }
                "interrupted" {
                    Run-Interrupted -Model $model -RunIndex $i -OutputDir $outputDir -TimeoutSeconds 15
                }
                "fragment_continue" {
                    Run-FragmentContinue -Model $model -RunIndex $i -OutputDir $outputDir -TimeoutSeconds 15
                }
            }
            
            # Small delay between runs to avoid rate limiting
            if (-not $DryRun) {
                Start-Sleep -Seconds 3
            }
        }
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  EXPERIMENT COMPLETE" -ForegroundColor Green
Write-Host "  Results in: $ResultsDir" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

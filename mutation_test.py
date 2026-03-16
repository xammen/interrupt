"""
Mutation Testing: Do more tests actually catch more bugs?

Strategy:
1. Define mutations (small code changes that introduce bugs)
2. Apply each mutation to the code
3. Run both 'complete' and 'two_step' test suites against mutated code
4. Compare: which test suite catches (kills) more mutants?

If two_step tests kill more mutants -> more tests = more reliable code detection
If same kill rate -> extra tests are redundant, no practical value
"""

import ast
import asyncio
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, stdev

RESULTS_DIR = Path("C:/Users/xammen/interrupt-study/results")

# ============================================================
# MUTATION DEFINITIONS
# ============================================================
# Each mutation is: (name, search_pattern, replacement)
# We apply regex mutations to the source code

MUTATIONS = [
    # --- Boundary mutations ---
    ("priority_lower_bound_off_by_one",
     r"not 1 <= self\.priority <= 10",
     "not 0 <= self.priority <= 10"),
    
    ("priority_upper_bound_off_by_one",
     r"not 1 <= self\.priority <= 10",
     "not 1 <= self.priority <= 11"),
    
    ("priority_check_removed",
     r"if not 1 <= self\.priority <= 10:.*?raise ValueError\([^)]+\)",
     "pass  # mutation: validation removed"),
    
    # --- Default value mutations ---
    ("default_max_retries_zero",
     r"max_retries:\s*int\s*=\s*3",
     "max_retries: int = 0"),
    
    ("default_status_running",
     r"status:\s*TaskStatus\s*=\s*TaskStatus\.PENDING",
     "status: TaskStatus = TaskStatus.RUNNING"),
    
    # --- Logic mutations ---
    ("circular_dep_check_disabled",
     r"(def _detect_cycle|def _check_circular|def _topological_sort|def detect_cycle)",
     "def _DISABLED_cycle_check_ORIGINAL"),
    
    ("retry_no_backoff",
     r"2\s*\*\*\s*(retry_count|attempt|retries|self\.retry_count|retry|i)",
     "1  # mutation: no exponential backoff, constant delay"),
    
    ("semaphore_unlimited",
     r"asyncio\.Semaphore\(\s*(\w+)\s*\)",
     "asyncio.Semaphore(999999)"),
    
    # --- Error handling mutations ---
    ("swallow_exceptions",
     r"raise (CircularDependencyError|DuplicateTaskError|MissingDependencyError)",
     "pass  # mutation: exception swallowed instead of "),
    
    ("duplicate_check_disabled",
     r"if\s+\w+\s+in\s+self\.\w+tasks\w*:",
     "if False:  # mutation: duplicate check disabled"),
    
    # --- Status mutations ---
    ("completed_not_set",
     r"(\.status\s*=\s*TaskStatus\.COMPLETED)",
     "pass  # mutation: status not set to COMPLETED"),
    
    ("failed_not_set",
     r"(\.status\s*=\s*TaskStatus\.FAILED)",
     "pass  # mutation: status not set to FAILED"),
    
    # --- Observer mutations ---
    ("observer_not_called",
     r"(await\s+\w*notify|await\s+\w*emit|await\s+\w*_fire|for\s+\w+\s+in\s+self\.\w*observers|for\s+\w+\s+in\s+self\.\w*listeners|self\._notify|self\.notify|self\.emit)",
     "pass  # mutation: observer notification disabled"),
    
    # --- Result mutations ---  
    ("result_not_stored",
     r"(\.result\s*=\s*)",
     "_ = "),
    
    # --- Metrics mutations ---
    ("start_time_not_recorded",
     r"(start_time\s*=\s*time\.)",
     "_ = time."),
    
    ("end_time_not_recorded",
     r"(end_time\s*=\s*time\.)",
     "_ = time."),
]


def apply_mutation(code: str, mutation_name: str, pattern: str, replacement: str) -> tuple[str, bool]:
    """Apply a mutation to code. Returns (mutated_code, was_applied)."""
    # Use re.DOTALL for multi-line patterns
    new_code, count = re.subn(pattern, replacement, code, count=1, flags=re.DOTALL)
    return new_code, count > 0


def run_tests_against_code(code_path: str, test_path: str, tmp_dir: str) -> dict:
    """Run a test file against a code file, return results."""
    # Copy both files to temp dir
    code_dest = Path(tmp_dir) / "task_scheduler.py"
    test_dest = Path(tmp_dir) / "test_task_scheduler.py"
    
    shutil.copy2(code_path, code_dest)
    shutil.copy2(test_path, test_dest)
    
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dest), "-v", "--tb=no", "-q", "--timeout=30"],
        capture_output=True, text=True, timeout=60,
        cwd=tmp_dir,
        env={**os.environ, "PYTHONPATH": tmp_dir}
    )
    
    # Parse results
    output = result.stdout + result.stderr
    
    # Count passed/failed
    passed = len(re.findall(r"PASSED", output))
    failed = len(re.findall(r"FAILED", output))
    errors = len(re.findall(r"ERROR", output))
    
    # Also check the summary line
    summary_match = re.search(r"(\d+) passed", output)
    if summary_match:
        passed = int(summary_match.group(1))
    
    fail_match = re.search(r"(\d+) failed", output)
    if fail_match:
        failed = int(fail_match.group(1))
    
    error_match = re.search(r"(\d+) error", output)
    if error_match:
        errors = int(error_match.group(1))
    
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": passed + failed + errors,
        "killed": failed + errors > 0,  # mutation was detected
        "returncode": result.returncode,
    }


def run_mutation_test(code_text: str, test_path: str, mutation_name: str, 
                      pattern: str, replacement: str, tmp_dir: str) -> dict:
    """Apply a mutation and run tests against it."""
    mutated_code, was_applied = apply_mutation(code_text, mutation_name, pattern, replacement)
    
    if not was_applied:
        return {"mutation": mutation_name, "applied": False, "killed": None}
    
    # Write mutated code
    code_dest = Path(tmp_dir) / "task_scheduler.py"
    code_dest.write_text(mutated_code, encoding="utf-8")
    
    # Copy test file
    test_dest = Path(tmp_dir) / "test_task_scheduler.py"
    shutil.copy2(test_path, test_dest)
    
    # Run pytest
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dest), "-x", "--tb=short", "-q", "--timeout=30"],
            capture_output=True, text=True, timeout=60,
            cwd=tmp_dir,
            env={**os.environ, "PYTHONPATH": tmp_dir}
        )
        
        output = result.stdout + result.stderr
        
        passed = 0
        failed = 0
        errors = 0
        
        summary_match = re.search(r"(\d+) passed", output)
        if summary_match:
            passed = int(summary_match.group(1))
        
        fail_match = re.search(r"(\d+) failed", output)
        if fail_match:
            failed = int(fail_match.group(1))
        
        error_match = re.search(r"(\d+) error", output)
        if error_match:
            errors = int(error_match.group(1))
        
        killed = result.returncode != 0
        
        # Extract which test caught it
        caught_by = re.findall(r"FAILED (test_\w+)", output)
        
        return {
            "mutation": mutation_name,
            "applied": True,
            "killed": killed,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "caught_by": caught_by[:3],  # first 3 tests that caught it
        }
    except subprocess.TimeoutExpired:
        return {
            "mutation": mutation_name,
            "applied": True,
            "killed": True,  # timeout = something broke = mutation detected
            "passed": 0, "failed": 0, "errors": 0,
            "caught_by": ["TIMEOUT"],
        }
    except Exception as e:
        return {
            "mutation": mutation_name,
            "applied": True,
            "killed": True,
            "passed": 0, "failed": 0, "errors": 0,
            "caught_by": [f"EXCEPTION: {e}"],
        }


def main():
    print("=" * 80)
    print("  MUTATION TESTING: Do more tests catch more bugs?")
    print("=" * 80)
    
    models = ["sonnet", "opus"]
    conditions = ["complete", "two_step", "code_given"]
    
    all_results = {}
    
    for model in models:
        all_results[model] = {}
        print(f"\n{'='*60}")
        print(f"  MODEL: {model.upper()}")
        print(f"{'='*60}")
        
        for run_idx in range(1, 6):
            print(f"\n  --- Run {run_idx} ---")
            
            # Get code from this run's complete condition
            code_path = RESULTS_DIR / f"{model}_complete" / f"run_{run_idx}_task_scheduler.py"
            code_text = code_path.read_text(encoding="utf-8")
            
            run_key = f"run_{run_idx}"
            all_results[model][run_key] = {}
            
            for condition in conditions:
                test_path = RESULTS_DIR / f"{model}_{condition}" / f"run_{run_idx}_test_task_scheduler.py"
                
                if not test_path.exists():
                    continue
                
                condition_results = []
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # First verify tests pass against unmutated code
                    baseline = run_mutation_test(
                        code_text, str(test_path), 
                        "BASELINE_NO_MUTATION", "THIS_WILL_NOT_MATCH", "x",
                        tmp_dir
                    )
                    
                    for mut_name, pattern, replacement in MUTATIONS:
                        result = run_mutation_test(
                            code_text, str(test_path),
                            mut_name, pattern, replacement,
                            tmp_dir
                        )
                        condition_results.append(result)
                
                applied = [r for r in condition_results if r.get("applied")]
                killed = [r for r in applied if r.get("killed")]
                survived = [r for r in applied if not r.get("killed")]
                
                kill_rate = len(killed) / len(applied) * 100 if applied else 0
                
                all_results[model][run_key][condition] = {
                    "total_mutations": len(MUTATIONS),
                    "applied": len(applied),
                    "killed": len(killed),
                    "survived": len(survived),
                    "kill_rate": round(kill_rate, 1),
                    "details": condition_results,
                }
                
                print(f"    {condition:>14}: {len(killed)}/{len(applied)} mutations killed ({kill_rate:.0f}%)")
    
    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print(f"\n\n{'='*80}")
    print("  AGGREGATE MUTATION KILL RATES")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\n--- {model.upper()} ---")
        print(f"{'Condition':<16} | {'Kill Rate':>10} | {'Killed':>8} | {'Applied':>8} | {'Per-run rates'}")
        print("-" * 75)
        
        for condition in conditions:
            rates = []
            killed_total = 0
            applied_total = 0
            
            for run_idx in range(1, 6):
                run_key = f"run_{run_idx}"
                if run_key in all_results[model] and condition in all_results[model][run_key]:
                    r = all_results[model][run_key][condition]
                    rates.append(r["kill_rate"])
                    killed_total += r["killed"]
                    applied_total += r["applied"]
            
            if rates:
                avg_rate = mean(rates)
                sd_rate = stdev(rates) if len(rates) > 1 else 0
                rates_str = ", ".join(f"{r:.0f}%" for r in rates)
                print(f"{condition:<16} | {avg_rate:>8.1f}% | {killed_total:>8} | {applied_total:>8} | {rates_str}")
    
    # Per-mutation analysis
    print(f"\n\n{'='*80}")
    print("  PER-MUTATION KILL ANALYSIS")
    print(f"{'='*80}\n")
    
    for model in models:
        print(f"\n--- {model.upper()} ---")
        print(f"{'Mutation':<35} | {'Complete':>10} | {'Two-Step':>10} | {'Code-Given':>10}")
        print("-" * 75)
        
        for mut_name, _, _ in MUTATIONS:
            row = f"{mut_name:<35}"
            for condition in conditions:
                kills = 0
                applies = 0
                for run_idx in range(1, 6):
                    run_key = f"run_{run_idx}"
                    if run_key in all_results[model] and condition in all_results[model][run_key]:
                        details = all_results[model][run_key][condition]["details"]
                        for d in details:
                            if d["mutation"] == mut_name:
                                if d.get("applied"):
                                    applies += 1
                                    if d.get("killed"):
                                        kills += 1
                row += f" | {kills}/{applies:>8}"
            print(row)
    
    # Save full results
    output_path = Path("C:/Users/xammen/interrupt-study/analysis/mutation_results.json")
    
    # Strip non-serializable data
    clean_results = {}
    for model in all_results:
        clean_results[model] = {}
        for run_key in all_results[model]:
            clean_results[model][run_key] = {}
            for cond in all_results[model][run_key]:
                r = all_results[model][run_key][cond]
                clean_results[model][run_key][cond] = {
                    k: v for k, v in r.items() if k != "details"
                }
                clean_results[model][run_key][cond]["per_mutation"] = [
                    {k: v for k, v in d.items()} for d in r["details"]
                ]
    
    with open(output_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    
    print(f"\n\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()

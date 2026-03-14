# The Interrupt Effect

**Does interrupting LLM code generation change the output? And if so, for better or worse?**

## TL;DR

We ran 30 experiments across Claude Sonnet 4.6 and Claude Opus 4.6, comparing three conditions: uninterrupted generation, interrupted + re-generated from scratch, and interrupted + continued from a 50% fragment. 

**Key finding:** When given their own partial code and asked to continue, LLMs produce **2x-4x more tests** with **qualitatively better test architecture** (testing the code, not the spec). The interruption doesn't degrade code quality — it redistributes the model's "attention budget" toward verification.

## The Experiment

### Task
Every run uses the exact same prompt: implement an async task scheduler in Python with dependency resolution, retry logic, observer pattern, and pytest tests. Medium complexity, ~350 lines of production code expected.

### Conditions

| Condition | What happens |
|-----------|-------------|
| **COMPLETE** | Full uninterrupted generation. Baseline. |
| **INTERRUPTED** | Generation killed at 20s, then fresh re-generation with the same prompt (no memory of the interruption). |
| **FRAGMENT** | Complete run cut at exactly 50% of lines, fragment fed back to the model with "continue from here + write tests". |

### Models
- Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`)
- Claude Opus 4.6 (`anthropic/claude-opus-4-6`)
- 5 runs per condition per model = **30 total runs**

### Tooling
All runs executed via [OpenCode CLI](https://opencode.ai) (`opencode run --model <model> <prompt>`) using the Anthropic provider with a Claude Max subscription (no API tokens consumed).

## Results

### Finding 1: Interruption + re-generation changes almost nothing

| Metric | Sonnet Complete | Sonnet Interrupted | Delta |
|--------|----------------|-------------------|-------|
| Lines of code | 371 | 351 | -5.5% |
| Functions | 15.4 | 14.2 | -7.8% |
| Feature score /10 | 9.0 | 9.0 | 0% |
| Test count | 22.0 | 20.2 | -8.2% |
| Syntax valid | 100% | 100% | = |

Same for Opus. **Killing and re-running produces statistically similar results.** The model doesn't "remember" being interrupted; it generates independently each time.

### Finding 2: Fragment+Continue produces 2x-4x more tests *(the big one)*

| | Sonnet Complete | Sonnet Fragment | Opus Complete | Opus Fragment |
|---|---|---|---|---|
| **Test functions** | 22.0 | **50.0** (+127%) | 10.4 | **42.8** (+311%) |
| **Assert statements** | 28.0 | **57.2** (+104%) | 25.2 | **67.0** (+166%) |
| **Test classes** | 5.0 | **12.0** | 0.0 | **13.4** |

This is **systematic across all 10 fragment runs without exception.** Every single fragment run produces at minimum 2x the tests of its complete equivalent.

#### But it's not just "more" — it's "better organized"

**Complete mode** organizes tests around the **prompt** (what was asked):
```
TestBasicExecution, TestDependencyResolution, TestCircularDependency, 
TestRetryLogic, TestConcurrencyLimit
```

**Fragment mode** organizes tests around the **code** (what exists):
```
TestTask, TestTaskMetrics, TestSchedulerMetrics, TestEventBus,
TestTaskSchedulerRegistration, TestDependencyValidation, TestExecutionPlan,
TestSchedulerRun, TestRetryLogic, TestEvents, TestConcurrency, TestEdgeCases
```

The fragment approach tests **the code**, not **the spec**. That's qualitatively better test engineering.

#### Granularity increase

For circular dependency testing:

**Complete:** 2 tests — "does it work" style
```
test_circular_dependency_detection
test_circular_dependency_three_node
```

**Fragment:** 5 tests — edge cases + error messages
```
test_self_loop
test_two_node_cycle  
test_three_node_cycle
test_error_message_contains_cycle_path
test_no_cycle_message
```

### Finding 3: Intra-condition variance differs by model

| | Complete | Interrupted | Fragment |
|---|---|---|---|
| **Sonnet** similarity | 0.17 | **0.34** | 0.15 |
| **Opus** similarity | **0.43** | 0.26 | 0.32 |

Interruption makes Sonnet **more consistent** but Opus **less consistent**. Inverse effect depending on the model.

### Finding 4: Feature completeness is invariant

9/10 required features present in every condition. The **what** doesn't change — models always implement what's asked. Only the **how** and **how much** vary.

### Finding 5: 100% syntax validity across all conditions

60 Python files generated, 0 syntax errors. Even fragment+continue (where the model receives code cut mid-method) produces valid Python every time.

## Why This Happens — The Mechanism

Three factors combine:

1. **Code review effect** — When the model receives existing code, it *reads* before it writes. This read-then-write mode activates more analytical behavior. It sees edge cases, branches, and error possibilities. In complete mode, it's in "continuous writing" mode without this step back.

2. **Attention budget redistribution** — The model has an implicit token/effort budget. In complete mode, it spends ~60% on code + ~40% on tests. In fragment mode, the code is "free" (already provided), so 100% of the budget goes to tests.

3. **Prompt displacement** — In complete mode, the model structures tests around the *prompt* (the 5 categories requested). In fragment mode, the prompt is far back in context, replaced by concrete code. The model structures tests around *the code*, which is objectively more rigorous.

## Practical Implications

- **For developers:** If you want better tests, **never ask the model to generate code AND tests in the same prompt**. Generate code first, then in a second call, give it the code and ask for tests separately. Our data shows this produces 2x-4x more tests with better structure.

- **For tool builders (Copilot, Cursor, OpenCode):** Consider implementing "staged generation" as a quality strategy — automatically splitting code+tests into two separate generation calls.

- **For alignment research:** This connects to context-dependent behavior. The quality/quantity of tests is not a fixed property of the model — it's a property of *when in the generation the tests arrive*.

## Connection to Prior Work

This finding mirrors the key.md discovery from [ai-test research](https://github.com/xammen/ai-test): **behavior is a property of context, not of the model**. With key.md, context made Llama 8B say "I don't know" instead of confabulating. Here, context (seeing its own code) makes Claude write 4x more tests with better architecture.

The principle is the same: *"Tell the model what it experienced, not what to do."*

## Repository Structure

```
interrupt-study/
  PROMPT.md                  # The exact prompt used for all runs
  analyze.py                 # Main analysis script
  analyze_results.py         # Alternative analysis (markdown extraction)
  run_experiment.ps1         # PowerShell experiment runner
  run_batch.ps1              # Batch runner script
  run_single.sh              # Single run bash script
  results/
    sonnet_complete/         # 5 runs: full generation
    sonnet_interrupted/      # 5 runs: killed + re-generated  
    sonnet_fragment_continue/# 5 runs: 50% fragment + continuation
    opus_complete/           # 5 runs: full generation
    opus_interrupted/        # 5 runs: killed + re-generated
    opus_fragment_continue/  # 5 runs: 50% fragment + continuation
  analysis/
    report.txt               # Generated analysis report
    full_analysis.json       # Raw metrics data
```

Each `results/<condition>/` directory contains:
- `run_N_task_scheduler.py` — The generated production code
- `run_N_test_task_scheduler.py` — The generated test code
- `run_N.txt` — Full opencode output
- `run_N_meta.json` — Run metadata (duration, etc.)
- `run_N_fragment.txt` — (fragment condition only) The 50% fragment used as input

## Reproducing

```bash
# Requires opencode with Anthropic provider configured
# Install: npm install -g opencode-ai
# Configure: opencode auth login (select Anthropic)

# Run all experiments
powershell -File run_batch.ps1

# Or run individual experiments
opencode run --model anthropic/claude-sonnet-4-6 "$(cat PROMPT.md)"

# Analyze results
python analyze.py
```

## Limitations

- N=5 per condition (30 total). Larger N would increase statistical power.
- Single task type (async scheduler). Different tasks may show different effects.
- Only Anthropic models tested. GPT-4o, Gemini, open-source models may behave differently.
- The "interrupt" in the INTERRUPTED condition is a timeout kill, not a user-initiated stop. The model doesn't know it was interrupted.
- Fragment cut point is exactly 50% — other cut points (25%, 75%) may produce different effects.

## License

MIT

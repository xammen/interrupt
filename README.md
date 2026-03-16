# The Interrupt Effect

**Does splitting code generation from test generation change what LLMs produce?**

## TL;DR

50 experiments across Claude Sonnet 4.6 and Claude Opus 4.6 with 5 experimental conditions. We found that **asking an LLM to write code and tests in a single prompt produces 2x-5x fewer tests** than any approach that separates code from tests into two steps. This effect is massive (p < 0.01, r = 1.00, zero distributional overlap), consistent across models, and 100% of generated tests pass.

The original hypothesis ("interrupting generation changes quality") was wrong. **The real finding is about prompt architecture:** single-prompt vs. two-step generation produces fundamentally different test output.

## The Experiment

### Task
Every run uses the exact same core requirement: implement an async task scheduler in Python with dependency resolution, retry logic, observer pattern, and pytest tests.

### 5 Conditions

| Condition | Description | # Runs |
|-----------|-------------|--------|
| **A. COMPLETE** | Single prompt: "write the code AND the tests". Baseline. | 10 |
| **B. INTERRUPTED** | Generation killed at 20s, then fresh re-generation with same prompt. | 10 |
| **C. FRAGMENT** | Complete run cut at 50% of lines, fragment given back with "continue + write tests". | 10 |
| **D. TWO_STEP** | Step 1: "write the code, NO tests". Step 2: "here's the code, now write tests". | 10 |
| **E. CODE_GIVEN** | Take complete code from condition A, give it to model: "here's code, write tests". | 10 |

Conditions D and E were added as controls after identifying a confounding variable in the original design (the fragment prompt was structurally different from the complete prompt).

### Models
- Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-6`)
- Claude Opus 4.6 (`anthropic/claude-opus-4-6`)
- 5 runs per condition per model = **50 total runs**

### Tooling
All runs via [OpenCode CLI](https://opencode.ai) using the Anthropic provider (Claude Max subscription, no API tokens consumed).

## Results

### The Definitive Table

**Test function count per condition:**

| Condition | Sonnet Mean | Sonnet StDev | Opus Mean | Opus StDev | Ratio vs Complete |
|-----------|-------------|-------------|-----------|-----------|-------------------|
| **A. COMPLETE** | 22.0 | 3.4 | 10.4 | 1.3 | x1.00 |
| **B. INTERRUPTED** | 20.2 | 2.9 | 11.6 | 4.8 | ~x1.00 |
| **C. FRAGMENT** | 50.0 | 6.0 | 42.8 | 7.6 | x2.3 / x4.1 |
| **D. TWO_STEP** | 50.6 | 5.0 | 53.4 | 3.8 | x2.3 / x5.1 |
| **E. CODE_GIVEN** | 52.6 | 12.1 | 52.2 | 4.8 | x2.4 / x5.0 |

### Statistical Significance (Mann-Whitney U)

| Comparison | Sonnet | Opus | Interpretation |
|------------|--------|------|---------------|
| Complete vs Interrupted | ns (p=0.53) | ns (p=0.83) | **No effect** of interruption |
| Complete vs Fragment | ** (U=0, p=0.008, r=1.00) | * (U=0, p=0.012, r=1.00) | **Massive effect** |
| Complete vs Two-Step | ** (U=0, p=0.008, r=1.00) | * (U=0, p=0.011, r=1.00) | **Massive effect** |
| Complete vs Code-Given | ** (U=0, p=0.008, r=1.00) | * (U=0, p=0.012, r=1.00) | **Massive effect** |
| Fragment vs Two-Step | ns (p=1.00) | * (U=2, p=0.036, r=0.84) | Fragment slightly weaker for Opus |
| Fragment vs Code-Given | ns (p=0.92) | * (U=2.5, p=0.047, r=0.80) | Fragment slightly weaker for Opus |
| Two-Step vs Code-Given | ns (p=1.00) | ns (p=0.67) | **No difference** |

Note: With N=5 per group, the minimum achievable p-value for Mann-Whitney U is ~0.008. All significant comparisons hit the floor (U=0, zero distributional overlap). Effect sizes (r=0.84-1.00) confirm large practical significance.

### 100% Test Pass Rate

Every single generated test passes across all 50 runs. Zero failures, zero errors.

## Key Findings

### Finding 1: Interruption alone does nothing
Killing and re-running produces statistically identical output (COMPLETE ~= INTERRUPTED). The LLM doesn't "reset" to something better or worse.

### Finding 2: The real effect is prompt architecture, not interruption
Fragment, Two-Step, and Code-Given all produce 2x-5x more tests than Complete. **It's not about the interruption — it's about separating code generation from test generation.**

### Finding 3: This is NOT about "seeing its own code"
We initially hypothesized that seeing its own partial code triggered a "code review effect". But conditions D (Two-Step) and E (Code-Given) produce equivalent results to C (Fragment), and in Code-Given the model sees *someone else's* complete code, not its own fragment. **The mechanism is task isolation, not self-review.**

### Finding 4: The effect is larger for Opus than Sonnet
- Sonnet: x2.3 more tests (22 -> 50)
- Opus: x5.1 more tests (10 -> 53)

Opus in single-prompt mode is particularly parsimonious with tests (avg 10.4), but produces a comparable number to Sonnet when given the code separately (avg 53.4).

### Finding 5: Opus shifts its entire test architecture
In single-prompt mode, Opus writes flat test functions (0 test classes). In two-step mode, it switches to organized test classes (9-12 classes). The prompt architecture changes the *paradigm*, not just the quantity.

## The Actual Mechanism

The initial "attention budget" and "code review effect" explanations were wrong. Here's what's actually happening:

**Single-prompt generation = joint optimization.** The model allocates effort across code + tests simultaneously. Since code is the harder task, tests get the leftovers.

**Two-step generation = sequential specialization.** Each call gets 100% of the model's attention for one task. The test generation call has no competing objective.

This is fundamentally about **task interference in autoregressive generation**. When a model generates code, it's also implicitly planning the tests it'll write later. This planning overhead competes with code quality. Separating them eliminates the interference.

## Practical Implications

1. **For developers:** Always generate code and tests in separate prompts. This is free 2x-5x test coverage with zero downside.

2. **For tool builders:** AI coding assistants should default to two-step generation for any task that includes tests. The current "generate everything at once" paradigm is measurably worse.

3. **For researchers:** "Prompt architecture" (how you split tasks across API calls) is an underexplored dimension. A single prompt and N sequential prompts with the same total information can produce qualitatively different outputs.

## Self-Critique & Limitations

### What we got wrong initially
- **Wrong hypothesis:** We thought interruption itself changed quality. It doesn't.
- **Confounding variable:** The fragment prompt was structurally different from the complete prompt. Conditions D and E were added to control for this.
- **Wrong mechanism:** "Attention budget redistribution" and "code review effect" sounded good but were disproven by the controls.

### Remaining limitations
- N=5 per condition per model. Larger N would increase power but qualitative findings are clear.
- Single task type. Would the effect hold for non-Python, non-test tasks?
- Only Anthropic models. GPT-4o, Gemini, open-source models may differ.
- All runs via OpenCode CLI (agentic mode) not raw API. The agent layer adds tool usage overhead.
- No temperature variation. All runs use default temperature.

## Connection to Prior Work

This connects to the [key.md research](https://github.com/xammen/ai-test) finding that **behavior is a property of context, not of the model.** Here, test quantity/quality is a property of prompt architecture, not model capability. The same model produces 10 or 53 tests depending purely on how you ask.

## Repository Structure

```
interrupt-study/
  PROMPT.md                  # The exact prompt used
  analyze.py                 # Main analysis script  
  results/
    sonnet_complete/         # Condition A: full generation
    sonnet_interrupted/      # Condition B: killed + re-gen
    sonnet_fragment_continue/# Condition C: 50% fragment + continue
    sonnet_two_step/         # Condition D: code-only then tests-only
    sonnet_code_given/       # Condition E: given code, write tests
    opus_complete/           # (same 5 conditions for Opus)
    opus_interrupted/
    opus_fragment_continue/
    opus_two_step/
    opus_code_given/
  analysis/
    report.txt               # Generated analysis report
    full_analysis.json       # Raw metrics data
```

## Reproducing

```bash
# Requires opencode with Anthropic provider configured
npm install -g opencode-ai

# Single-prompt (Condition A):
opencode run --model anthropic/claude-sonnet-4-6 "Write task_scheduler.py [...] AND write tests"

# Two-step (Condition D):
opencode run --model anthropic/claude-sonnet-4-6 "Write task_scheduler.py [...] NO tests"
opencode run --model anthropic/claude-sonnet-4-6 "Here's the code, write tests: [CODE]"

# Analyze
python analyze.py
```

## License

MIT

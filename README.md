# The Interrupt Effect

**Does splitting code generation from test generation change what LLMs produce?**

## TL;DR

50 experiments across Claude Sonnet 4.6 and Claude Opus 4.6 with 5 experimental conditions. We found that **asking an LLM to write code and tests in a single prompt produces 2x-5x fewer tests** than any approach that separates code from tests into two steps. Mutation testing proves these aren't redundant tests: **single-prompt tests miss 1 in 3 injected bugs, while two-step tests catch 100%.**

The deeper finding: it's not about task separation — it's about the **curse of knowledge**. The model that just wrote `max_retries: int = 3` never tests that the default is actually 3. It "knows" what it wrote, so it doesn't verify it. A separate test-generation call approaches the code with fresh eyes and questions everything.

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

### Test Quantity

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

### Mutation Testing: Do More Tests Actually Catch More Bugs?

We injected 16 mutations (bugs) into the code — boundary errors, disabled checks, swallowed exceptions, wrong defaults — and ran each condition's test suite against the mutated code.

**Mutation kill rates:**

| Condition | Sonnet Kill Rate | Opus Kill Rate |
|-----------|-----------------|---------------|
| **A. COMPLETE** | 63.5% | 66.7% |
| **D. TWO_STEP** | **100%** | **100%** |
| **E. CODE_GIVEN** | 87.2% | 96.7% |

**Single-prompt tests let 1 in 3 bugs escape. Two-step tests catch every single one.**

The bugs that survive single-prompt testing are revealing:

| Mutation | Complete catches | Two-Step catches |
|----------|-----------------|-----------------|
| `max_retries` default changed 3→0 | **0/10** (never) | **10/10** (always) |
| `status` default changed PENDING→RUNNING | **0/10** (never) | **10/10** (always) |
| Priority boundary off-by-one | 4/10 | **10/10** (always) |

The model that just wrote `max_retries: int = 3` never verifies the default is 3. It "knows" what it wrote.

### Code Quality: No Difference

The code itself is statistically identical across all conditions (all p > 0.15):

| Metric | Complete | Two-Step | Difference |
|--------|----------|----------|-----------|
| Lines of code | 367 | 391 | ns |
| Functions | 14.9 | 16.1 | ns |
| Classes | 6.6 | 6.6 | ns |
| Feature score | 9.0/10 | 9.0/10 | identical |

The model doesn't write better code when freed from test planning. It writes the same code, then writes dramatically better tests separately.

### Token Cost

| | Sonnet | Opus |
|---|---|---|
| Complete: total tokens | ~8,100 | ~6,900 |
| Two-Step: total tokens | ~15,900 | ~16,500 |
| **Token overhead** | **+95%** | **+140%** |
| **Test increase** | **+130%** | **+413%** |
| **Mutation kill increase** | **+36pp** (64%→100%) | **+33pp** (67%→100%) |
| Tests per 1k tokens | 2.70 → 3.19 | 1.51 → 3.23 |

Two-step costs roughly 2x in tokens but produces 2-5x more tests that catch 50% more bugs. **Per-token, two-step is more efficient, not less** — especially for Opus (2.1x more tests per token).

### What Tests Are Missing in Single-Prompt?

It's not that two-step tests go deeper on the same concepts. The **depth per concept is identical** (~1.1 tests/concept in both conditions). The difference is purely in **breadth** — how many distinct concepts the model thinks to test:

| | Complete | Two-Step |
|---|---|---|
| **Sonnet** | 20.8 concepts × 1.06 depth | 42.4 concepts × 1.20 depth |
| **Opus** | 9.6 concepts × 1.08 depth | 47.8 concepts × 1.12 depth |

The categories that explode in two-step:

| Category | Sonnet ratio | Opus ratio |
|----------|-------------|-----------|
| **Unit property tests** (defaults, boundaries) | **x7.6** | **x11.0** |
| **Edge cases** (errors, empty, null) | x1.7 | **x7.7** |
| **Metrics tests** | x1.2 | **∞** (0→3/run) |

Single-prompt Opus writes 0 tests for metrics, 0 for default values, and 0.8 for edge cases. It focuses exclusively on the "interesting" behaviors (dependency resolution, retry logic, concurrency). The "boring" stuff — does the default actually work? — gets skipped entirely.

## Key Findings

### Finding 1: Interruption alone does nothing
Killing and re-running produces statistically identical output (COMPLETE ≈ INTERRUPTED). The LLM doesn't "reset" to something better or worse.

### Finding 2: Task separation produces 2-5x more tests that catch real bugs
Fragment, Two-Step, and Code-Given all produce massively more tests than Complete. These aren't padding — mutation testing confirms they catch bugs that single-prompt tests miss (100% vs ~65% kill rate).

### Finding 3: The mechanism is the curse of knowledge
The model that wrote `max_retries: int = 3` never tests that default. The model that wrote circular dependency detection never tests boundary cases of the detection. **It skips testing what it considers "obvious" because it just implemented it.** A separate call doesn't have this implicit knowledge and questions everything — including defaults, boundaries, and edge cases.

This is not "task isolation" in the abstract. It's specifically that **the author's knowledge of their own implementation creates blind spots in testing**. The same cognitive bias is well-documented in human software engineering.

### Finding 4: The effect is larger for Opus than Sonnet
- Sonnet: x2.3 more tests, kill rate 64%→100%
- Opus: x5.1 more tests, kill rate 67%→100%

Opus in single-prompt mode is particularly opinionated — it writes only 10.4 tests focused on "interesting" behaviors. Given the code separately, it writes 53.4 tests covering everything. Stronger priors = stronger curse of knowledge.

### Finding 5: Two-step is more token-efficient, not less
Despite costing ~2x in raw tokens, two-step produces 2-5x more tests. Per-token test yield is 1.2-2.1x higher. And since those tests catch 50% more mutations, the cost-per-bug-caught is dramatically lower.

## The Actual Mechanism

### What we initially thought (wrong)
- ~~"Attention budget redistribution"~~ — Disproven: code quality is identical
- ~~"Code review effect"~~ — Disproven: Code-Given (seeing someone else's code) works equally well
- ~~"Task isolation"~~ — Partially right, but misses the deeper cause

### What's actually happening

**The curse of knowledge in autoregressive generation.**

When a model generates code, it builds an implicit mental model of "what this code does" and "what's important about it." When it then generates tests in the same context, this mental model acts as a filter: it tests the "interesting" parts (dependency resolution, retry logic) and skips the "obvious" parts (do defaults work? are boundaries correct?).

In a separate call, the model has no such mental model. It reads the code as an external artifact and systematically tests everything — including the things the author would consider too trivial to verify.

This is the exact same bias that makes human developers bad at testing their own code:
- **"I just wrote it, I know it works"** → doesn't test defaults
- **"This boundary check is straightforward"** → doesn't test off-by-one
- **"The observer pattern is the tricky part"** → tests only the complex bits

The LLM exhibits the same bias, and the fix is the same: **have someone else write the tests.**

## Practical Implications

1. **For developers:** Always generate code and tests in separate prompts. It costs 2x tokens but produces 2-5x more tests that catch 50% more bugs. The per-token efficiency is actually higher.

2. **For tool builders:** AI coding assistants should default to two-step generation for any task that includes tests. The current "generate everything at once" paradigm produces tests with 35% mutation survival rate.

3. **For researchers:** LLMs exhibit the curse of knowledge — the same cognitive bias that affects human developers. This has implications beyond testing: any task where an LLM generates an artifact and then evaluates/validates it in the same context may suffer from the same blind spot.

4. **For prompt engineers:** "Prompt architecture" (how you split tasks across API calls) is an underexplored dimension with measurable quality impact. A single prompt and N sequential prompts with the same total information produce qualitatively different outputs.

## Self-Critique & Limitations

### What we got wrong along the way
- **Wrong hypothesis:** We thought interruption itself changed quality. It doesn't.
- **Confounding variable:** The fragment prompt was structurally different from the complete prompt. Conditions D and E were added to control for this.
- **Wrong mechanism (twice):** "Attention budget redistribution" and "code review effect" sounded good but were disproven by the controls. "Task isolation" was closer but missed the curse-of-knowledge angle.

### Remaining limitations
- N=5 per condition per model. Larger N would increase statistical power but qualitative findings are unambiguous.
- Single task type. Would the effect hold for non-Python, non-test tasks?
- Only Anthropic models. GPT-4o, Gemini, open-source models may differ.
- All runs via OpenCode CLI (agentic mode) not raw API. The agent layer adds tool usage overhead.
- No temperature variation. All runs use default temperature.
- Mutation set is hand-crafted (16 mutations). A larger automated mutation set would be more rigorous.

## Connection to Prior Work

This connects to the [key.md research](https://github.com/xammen/ai-test) finding that **behavior is a property of context, not of the model.** Here, test quantity/quality is a property of prompt architecture, not model capability. The same model produces 10 or 53 tests depending purely on how you ask.

The curse of knowledge connection also links to established software engineering research on why code review and independent testing catch bugs that the original author misses. LLMs appear to exhibit the same cognitive pattern.

## Repository Structure

```
interrupt-study/
  PROMPT.md                  # The exact prompt used
  analyze.py                 # Main analysis script (5-condition, Mann-Whitney U)
  mutation_test.py           # Mutation testing framework
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
    mutation_results.json    # Mutation testing results
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
python mutation_test.py
```

## License

MIT

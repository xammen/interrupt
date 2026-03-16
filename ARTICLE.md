# I ran 50 experiments interrupting Claude mid-generation. Interruption isn't the story.

The way you ask changes everything. Not the model. Not the prompt content. Just whether you ask for code and tests in one shot, or two separate calls.

Here's what I found.

---

## The setup

Same task every time: write an async task scheduler in Python. Priority queues, dependency resolution, retry logic, observer pattern. Write the code. Write the tests.

I ran this 50 times across Claude Sonnet 4.6 and Opus 4.6, with 5 different conditions:

- **Complete**: one prompt, code + tests together
- **Interrupted**: killed the generation at 20s, re-ran the same prompt fresh
- **Fragment**: took a complete run, cut it at 50% of lines, fed the fragment back with "continue and add tests"
- **Two-step**: first prompt asks for code only, no tests. Second prompt: here's the code, now write the tests.
- **Code-given**: took the complete code from condition A, gave it to the model cold: here's the code, write tests.

5 runs per condition per model. 50 total. All via OpenCode CLI on Claude Max so no API cost.

---

## Finding 1: Interrupting does nothing

Killing and restarting produces statistically identical output (p=0.53 for Sonnet, p=0.83 for Opus). The model doesn't reset to something better. No fresh start effect. Same code, same number of tests, same everything.

The title of this post is a lie. Interruption is not the story.

---

## Finding 2: The number of tests explodes when you separate the tasks

| Condition | Sonnet | Opus |
|-----------|--------|------|
| Complete (one prompt) | 22 tests | 10 tests |
| Two-step | 51 tests | 53 tests |
| Code-given | 53 tests | 52 tests |

Sonnet goes from 22 to 51. Opus goes from 10 to 53. Zero distributional overlap between conditions (U=0, r=1.00). This is not noise.

The code itself? Identical across conditions. Same lines, same functions, same classes, same feature score. The model doesn't write better code when freed from test planning. It writes the same code, and then dramatically better tests.

---

## Finding 3: The extra tests are not padding

This is the part I didn't expect.

I ran mutation testing. Injected 16 bugs into the code: off-by-one boundary errors, wrong default values, disabled exception handling, swallowed observers. Then ran both test suites against each mutated version.

- Complete tests: **65% mutation kill rate**
- Two-step tests: **100% mutation kill rate**

Single-prompt tests let 1 in 3 bugs through. Every time.

The bugs that survive are the same ones, run after run:

| Bug injected | Complete catches | Two-step catches |
|---|---|---|
| `max_retries` default changed 3 to 0 | 0 out of 10 | 10 out of 10 |
| `status` default changed PENDING to RUNNING | 0 out of 10 | 10 out of 10 |
| Priority boundary off-by-one | 4 out of 10 | 10 out of 10 |

---

## Why

The model that just wrote `max_retries: int = 3` never tests that the default is actually 3. It knows what it wrote. It doesn't think to verify it.

I looked at the test names across conditions. The difference isn't depth, it's breadth. Both conditions write about 1.1 tests per concept they choose to test. But two-step identifies 42 concepts worth testing. Single-prompt identifies 20.

The categories that vanish in single-prompt mode:

- **Unit property tests** (do the defaults work, are the boundaries correct): 7x fewer in Sonnet, 11x fewer in Opus
- **Edge cases**: 1.7x to 7.7x fewer depending on model
- **Metrics tests**: Opus writes exactly 0 metrics tests in single-prompt mode

The model focuses on what it considers interesting: dependency resolution, retry logic, concurrency. The "obvious" stuff, the defaults, the boundaries, gets skipped. Because it just wrote them. It knows they work.

This is the curse of knowledge. The same reason human developers are bad at testing their own code. You wrote it, so you have blind spots about it. The fix for humans is code review. The fix for LLMs turns out to be the same thing: have someone else write the tests.

In Code-Given condition, I gave the model someone else's complete code and asked it to write tests. It produced 52 tests with an 87-97% mutation kill rate. Nearly identical to two-step. The model doesn't need to have written the tests in a separate session to gain fresh eyes. It just needs to not have written the code in the same session.

---

## The token question

Two-step costs roughly 2x in tokens (you're making two full API calls, and the second call has the code as input context).

| | Sonnet | Opus |
|---|---|---|
| Complete | ~8,100 tokens | ~6,900 tokens |
| Two-step | ~15,900 tokens | ~16,500 tokens |
| Overhead | +95% | +140% |

But per-token, two-step is actually more efficient:

- Sonnet: 2.70 tests per 1k tokens single-prompt, 3.19 two-step
- Opus: 1.51 tests per 1k tokens single-prompt, 3.23 two-step

You pay 2x in tokens, get 2-5x more tests, and those tests catch 50% more bugs. The cost per bug caught is lower in two-step, not higher.

---

## What this means practically

If you use AI to write code with tests, always use two prompts. First ask for the code. Then paste the code and ask for tests separately. It takes 30 extra seconds and produces a test suite that catches everything instead of 65% of things.

The current default behavior of AI coding tools, where you describe a feature and get code and tests in one response, is measurably worse. The test suite it produces has a 35% mutation survival rate. A third of bugs make it through.

---

## What I got wrong along the way

My original hypothesis was that interrupting generation forces a "fresh start" that produces better output. That's wrong. Interruption does nothing.

My second hypothesis was a "code review effect": seeing its own partial code made the model examine it critically. Also wrong. Code-given uses someone else's complete code and gets the same result.

My third hypothesis was "task isolation" in the abstract. Closer, but it misses the actual mechanism.

The real answer is simpler and more human: the model has blind spots about code it just wrote. The same blind spots developers have about code they just wrote. And the fix is the same one we've known about for decades in software engineering. Don't test your own code.

---

Full data, analysis scripts, and all 50 generated files: [github.com/xammen/interrupt](https://github.com/xammen/interrupt)

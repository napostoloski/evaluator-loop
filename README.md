# evaluator-loop

**The evaluator loop: an AI model proposes what drives each evaluator's judgments and writes
one-detail edits of real applications; re-ratings confirm what holds.**

Research artifact for the Organization Science special issue proposal *From Prediction to
Explanation: Using AI to Design Counterfactual Experiments of Human Judgment*. The method is
built for panels in which every evaluator scores the same applications, such as the 156
startup applications to an accelerator program in Pakistan that 20 evaluators scored in early
2026.

The repository asks **which feature of an application drives a particular evaluator's
judgment, beyond what the whole panel reacts to**. It contains the prototype, the benchmark
behind Table 1 of the proposal, in which the evaluators' preferences are planted so the right
answer is known, the three runs with a commercial language model, a local app (including a
path for researchers' own data) and a web explorer.

No real application, rating or outcome is included. Everything here is synthetic.

## For reviewers

- **Interactive explorer** (no installation): [the evaluator loop explorer](https://claude.ai/artifact/5PC1tA5siacgEBm1G27Mhn).
  Choose the settings, then click any of the 20 simulated evaluators to see the hypotheses the
  prototype tested and whether each finding was real. The same page is in `explorer/`.
- **Specification:** [`docs/technical_note.md`](docs/technical_note.md), the basis of the
  technical appendix.
- **Every benchmark table:** [`outputs/REPORT.md`](outputs/REPORT.md).
- **Runs with a language model:** [`outputs/llm_sonnet/RUN_SUMMARY.md`](outputs/llm_sonnet/RUN_SUMMARY.md),
  with each run's audit trail and every saved model call in `outputs/llm_sonnet*/`.

**Terms.** The proposal says *preference* where the code and report say *rule*, and
*statistical copy* where they say *twin*.

### Where each number in the proposal comes from

Benchmark numbers are means over 12 simulated panels (seeds 100 to 111) with half-point
preferences, 30 re-ratings per hypothesis, up to 150 per evaluator, and re-rating consistency
ρ = 0.6. In `outputs/summary_recovery.csv` these are the rows with `delta` 0.5, `budget` 150,
`battery_apps` 30 and `rho` 0.6.

| Proposal | Value | Source |
|---|---|---|
| Table 1, prototype | 10.2 of 16 found (64%), 90% real, 7% wrongly flagged | method `loop`: recall 0.635, precision 0.904, null rate 0.069 |
| Table 1, LASSO candidates, untested | 12.3 (77%), 51%, 39% | method `screen`: 0.771, 0.505, 0.389 |
| Table 1, LASSO candidates with the statistical copy, no re-ratings | 14.3 (89%), 39%, 67% | method `twin`: 0.891, 0.387, 0.667. It finds more than `screen` because it accepts on the copy's estimate rather than the LASSO's and runs a second round |
| Table 1, regression on all ten candidate features | 7.2 (45%), 80%, 13% | method `library`: 0.448, 0.804, 0.125 |
| Introduction, 90% against 51% real | | Table 1, first two rows |
| Re-ratings spread evenly over all ten features | 5% found | method `edit_all`: recall 0.052 |
| Same-gender premium: 85 of 240 evaluators; source identified in 82% | against 62% for the statistical copy alone, 61% and 53% for the regressions | `outputs/summary_root_attribution.csv`, `delta` 0.5, column `names_the_right_feature` (methods `loop`, `twin`, `coded`, `library`) |
| Smaller preferences | 8% found at a quarter point, 86% at three quarters | method `loop`, `delta` 0.25 and 0.75 |
| Runs with Claude Sonnet 5.5 | 1, then 6 and 7 of 16 found, all real, none for the six evaluators without a preference; runs 2 and 3 about an hour and $8.49 and $10.90 | [`outputs/llm_sonnet/RUN_SUMMARY.md`](outputs/llm_sonnet/RUN_SUMMARY.md) |

### Reproducing the results

```bash
pip install -r requirements.txt
python -m pytest                          # the tests, about a minute
python -m evloop benchmark --seeds 12     # the full benchmark behind Table 1 (about 30 minutes on 2 cores)
python -m evloop report                   # rebuilds outputs/REPORT.md and the summary files
```

The third run with a language model replays exactly from its saved calls, with no API key and
no model calls:

```bash
mkdir -p replay && tar -xzf outputs/llm_sonnet_v2/llm_cache.tar.gz -C replay
python -m evloop llm-demo --base-url https://api.anthropic.com/v1 --model claude-sonnet-5-5 \
    --temperature -1 --seed 1 --extra-body '{"thinking": {"type": "between_tools"}}' \
    --twin ridge --twin-pairs 20 --workers 4 --proposer-prompt v2 --hypotheses 6 --examples 20 \
    --out replay
```

It takes about a minute, ends with "model calls: 0" and writes the same audit trail as
`outputs/llm_sonnet_v2/llm_demo_seed1_audit_trail.csv` (the current code adds one column,
`n_pairs`): every hypothesis, screen and test is identical, including the seven confirmed
preferences and their estimates. The first two runs used earlier versions of the code, so their saved
calls are an audit record rather than an exact replay.

## The idea

A human evaluator cannot be re-run on the same application with one cue changed. A model of
that evaluator can. The loop runs in rounds over the whole panel:

1. **Observe.** Take the part of each evaluator's scores that is their own: the residual after
   regressing their scores on the other evaluators' consensus.
2. **Propose.** From the text alone, generate hypotheses about which features of the
   applications explain that residual.
3. **Screen.** Test each hypothesis on a twin of the evaluator: edit the applications and see
   whether the twin's predicted scores move.
4. **Confirm.** Run a re-rating battery. Applications are edited so that the feature switches,
   and every evaluator re-rates the edited versions, with each evaluator's original rating as
   their own control. A rule is confirmed for an evaluator only if that evaluator's reaction
   differs from the rest of the panel's. Reactions everyone shares are not reported as anyone's
   rule. False discoveries are controlled panel-wide.

Confirmed rules are subtracted and the loop repeats. The result for each evaluator is a short
list of rules, for example "this evaluator scores founders of their own gender about 0.5
points higher than the panel does, triggered by the founder's name and pronouns". Each rule
comes with the evidence trail that supports it.

## Why a planted-rule benchmark

Real applications cannot be released, and real evaluators' rules are unknown. Both make it
impossible to check whether a loop like this finds true sources or produces plausible
stories. The benchmark builds a world where the truth is known:

- **Synthetic applications** are rendered from attributes, with confounds built in. Women
  founders are more often in consumer sectors and run home-based businesses. Large-city
  founders write more fluent English.
- **Synthetic evaluators** all share some reactions (to paying customers, fluency, experience
  and solo founders). Fourteen of the twenty also carry planted rules: a same-gender cue,
  gender-correlated sector and home-based proxies, an extra fluency premium, an informative
  "paying customers" premium, and a combination. Six carry none.
- **Single-feature edits are exact.** Re-rendering with one attribute changed alters only the
  sentences that express it.

A loop that cannot recover planted rules here, or that reports shared reactions as
individual ones, should not be trusted on real evaluators.

## Results

Twelve replications per cell, with benchmark seeds disjoint from the development seeds. The
world is deliberately noisy:

- evaluators agree with each other at r ≈ 0.4;
- every evaluator shares some reactions, which vary a little between evaluators;
- a re-rating months later correlates only about 0.7 with the original rating.

With planted rules of 0.5 points on the 1-5 scale and 150 re-ratings per evaluator:

| Method | Precision | Recall | Null evaluators given a rule |
|---|---|---|---|
| **Full loop** (propose, screen, re-rate 5 features × 30 applications) | **0.90** | **0.64** | 7% |
| Loop testing total rather than idiosyncratic reactions | 0.87 | 0.66 | 14% |
| Edit everything: same 150 re-ratings over all ten features, no screen | 0.91 | 0.05 | 0% |
| Twin only (no re-ratings) | 0.39 | 0.89 | 67% |
| Text screen only | 0.51 | 0.77 | 39% |
| Regression on all ten coded features (no re-ratings) | 0.80 | 0.45 | 13% |
| Regression on the usual structured covariates | 0.74 | 0.33 | 11% |

![Recovery by effect size](outputs/fig_recovery_by_effect_size.png)

What the benchmark shows:

- **The loop confirms most half-point rules, and few false ones.** It recovers 64% of planted
  rules with 90% precision, against 45% and 80% for the best method without edits. At 0.75
  points it recovers 86%.
- **Screening is what makes confirmation affordable.** Under realistic re-rating noise, an
  individual evaluator's rule needs about 30 re-ratings per tested feature. With 156
  applications, each re-rated at most once, spreading the budget over all ten features leaves
  15 per feature, and almost nothing is confirmed (recall 0.05). Screening concentrates the
  budget on the few features worth testing.
- **The real re-ratings are what make it precise.** The twin alone accepts many false rules
  (precision 0.39).
- **Testing against the panel halves false rules for null evaluators,** from 14% to 7%.
- **It locates the source of an apparent same-gender premium.** Among evaluators with a
  significant same-gender premium, the loop classifies it correctly as the name-and-pronoun
  cue or as gender-correlated content (sector or home-based) in 82% of cases, and names the
  right feature in those cases. Regressions without edits name the right feature in 53% to
  61%. The twin alone classifies cue against content correctly in 86% but names the right
  feature in only 62%. A redaction battery with the same budget separates cue from content
  in 66%.

Where it is weak:

- **Small rules cannot be confirmed for one evaluator at this budget.** At 0.25 points,
  recall is 8%. Rules that small need pooling across evaluators, for example all women
  evaluators against all men.
- **Some false positives remain (7% of null evaluators).** Part of the cause is regression to
  the mean: the hypothesis was chosen on the same evaluator's noisy original scores.
  - Two stricter tests are included: one allows for ordinary differences between evaluators,
    the other adjusts for the original residual. Each raises precision to 0.95 or 0.96 and
    cuts recall to 0.47 or 0.50.
  - Which trade-off to use is a design choice for Stage 2.
- **Writing-style rules are hardest to find** (recall 0.42).
- **Results depend on how noisy re-ratings are.** If re-ratings repeat the first reading more
  closely, recall rises (0.74 at ρ = 0.9, against 0.51 at ρ = 0.3), but so do false rules
  (17% against 4% of null evaluators).
- **The outcome test cannot tell ability from bias at this sample size.** With 156 startups it
  labels 14% of the truly informative rules as ability, against 7% of the uninformative ones.

**Implication for Stage 2:** plan on about 30 re-ratings per tested feature per evaluator,
for roughly the five features the screen ranks highest. That is about 150 re-ratings each,
similar to the original workload. Use pooled tests for effects below half a point.

Every table is in [`outputs/REPORT.md`](outputs/REPORT.md). The full specification is in
[`docs/technical_note.md`](docs/technical_note.md).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                          # 30 tests, about a minute
python -m evloop demo --seed 1            # one synthetic world: per-evaluator results and the audit trail
python -m evloop benchmark --seeds 12     # the full benchmark (about 30 minutes on 2 cores)
python -m evloop report                   # rebuild tables and figures from outputs/benchmark_rows.csv
```

## The app: settings, your own data, results per evaluator

```bash
pip install -e ".[app]"
python -m evloop app                      # opens http://localhost:8501
```

**Simulated panel.** Choose the size of the planted preferences, how consistent re-ratings are,
how many re-ratings each hypothesis gets and which simulated panel to draw; each setting is
explained on the page, as are how the synthetic applications and evaluators are made. Run the
prototype offline (about 20 seconds) or with an AI model, then see every evaluator's planted
preferences next to the hypotheses the prototype considered: what the statistical copy
predicted, what the re-ratings showed, and whether each confirmed preference was real.

**Your own data.** Upload a panel where every evaluator rated the same applications:

| file | columns |
|---|---|
| `applications.csv` | `application_id`, `text`, `founder_gender` (optional, F or M) |
| `ratings.csv` | `evaluator_id`, `application_id`, `score` (one row per rating) |
| `evaluators.csv` | `evaluator_id`, `gender` (optional); the whole file is optional |

Real data has no re-ratings yet, so the prototype stops after screening. For each evaluator it
returns the hypotheses an AI model proposed, whether the statistical copy supports them,
example edits, and, for the most requested features, the edited applications a re-rating study
would use (`hypotheses.csv`, `edits_to_rerate.csv`). Nothing is confirmed until evaluators
re-rate. A quick look without any model shows the words most associated with each evaluator's
departures from the panel. Applications are sent only to the AI provider chosen in the sidebar;
check that the data permissions allow it. The app can write example files from a simulated
panel to try the upload path first. The same path is available in code:
`evloop.realdata.load_dataset` and `evloop.realdata.screen_dataset`.

**Web explorer.** `explorer/index.html` is a page that shows pre-computed offline runs for
every combination of the settings above (3 preference sizes x 3 re-rating consistencies x 3
battery sizes x 3 panels = 81 runs) and the real-AI run, with each evaluator's hypotheses
against the planted truth. Serve the folder (`python -m http.server -d explorer`) and open it.
To rebuild its data: `python scripts/precompute_runs.py` (about 30 minutes on 2 cores), then
`python scripts/build_explorer_data.py outputs/interface explorer/runs.json`.

## Running the loop with an LLM

The offline backend uses a sparse text model as the proposer and a ridge text model as the
twin, so the benchmark runs anywhere without model access. `evloop/llm.py` provides an LLM
backend with the same interface:

- a proposer that states hypotheses in plain language;
- a minimal-edit rewriter, with a word-level similarity check and a manipulation check;
- a feature coder;
- a few-shot twin of each evaluator's deviations from the panel.

It talks to any OpenAI-compatible server using only the standard library. Replies are
validated before they are cached, failed requests are retried, and every call is logged to
JSONL so a run can be audited and replayed.

The three runs so far used Claude Sonnet 5.5 (below). The same client works with an
open-weight model you host yourself, for example with vLLM; this path has not yet been run
end to end:

```bash
vllm serve <model>
python -m evloop llm-demo --base-url http://localhost:8000/v1 --model <model> --no-thinking
```

`--no-thinking` turns off the reasoning mode that some open-weight models use by default when
served by vLLM. Other servers may ignore it. Reasoning blocks are stripped from every reply.
If a model spends its whole token limit reasoning, the client stops with an error rather than
continuing silently.

Useful flags: `--twin ridge` (screen with the offline ridge copy on the model's edits, no
twin calls), `--twin-pairs 30` (edited pairs per hypothesis), `--no-verify-edits` (skip
the verifier call on battery edits), `--extra-body '<json>'` (merged into every request).

Commercial endpoints work the same way (set `EVLOOP_API_KEY`). For models that reject
`temperature` or `max_tokens`, pass `--temperature -1 --max-tokens-field max_completion_tokens`.
`--temperature` and `--top-p` set sampling. Replies are cached, so runs stay reproducible at
any temperature.

Claude models reason before they answer and the OpenAI-style endpoint does not return that
reasoning, so turn it off and leave the temperature unset:

```bash
EVLOOP_API_KEY=... python -m evloop llm-demo --base-url https://api.anthropic.com/v1 \
    --model claude-sonnet-5-5 --temperature -1 --extra-body '{"thinking": {"type": "between_tools"}}'
```

### Runs with Claude Sonnet 5.5

The LLM path was first run end to end on Claude Sonnet 5.5 on 23 October 2026 (seed 1, about
4,300 live calls, $21 over 3.5 hours). The model proposed four plain-language hypotheses per
evaluator, rewrote the applications, coded features and acted as every evaluator's copy;
the simulated evaluators re-rated the edits as in the benchmark.

- 80 hypotheses proposed, 27 passed the copy's screen, 5 reached a re-rating battery,
  1 was confirmed, and it was correct: Evaluator 0's planted same-gender cue, estimated
  at +0.45 against the panel (planted +0.50), q = 0.0004. No false confirmations.
- Recall was 1 of 16 planted preferences. The cause is not the confirmation step but three
  weaknesses of the AI steps that the offline stand-ins do not have: free-text hypotheses
  were not pooled across evaluators, so batteries went to singletons and 22 passing
  hypotheses were never tested; the language-model copy, predicting from 10 pairs, is
  noisier than the ridge copy; and the rewrites were often half-edits (pronouns switched,
  name kept) that the checks let through.
- All three are now fixed in the code, and tested without further model calls:
  - **Pooling.** After screening, one model call groups the passing hypotheses into
    distinct features with a canonical wording and a polarity, so "the founder is a
    man" and "the founder is a woman" share a battery with opposite directions.
  - **Screening.** 30 edited pairs per hypothesis instead of 10, built from one rewrite
    per application rather than two. `--twin ridge` screens the model's edits with the
    offline ridge copy instead of a few-shot model copy, which removes the most expensive
    calls (about 40% of the first run) and, in the benchmark, is the more reliable filter.
  - **Edits.** Free checks reject rewrites that change the sections, mix pronouns, switch
    pronouns without the name, or come back unchanged, and strip appended commentary.
    Battery edits also pass a verifier call that compares original and rewrite.
    `python -m evloop audit-edits --log outputs/llm_sonnet/llm_calls.jsonl.gz` replays
    these checks over the first run's 1,822 rewrites: among rewrites where a change was
    needed, the old checks accepted 463 of 465 and 71% of those switched the detector;
    the new checks accept 393 and 84% switch it (gender edits: 100%, up from 51%).
  - On the stub server, the full demo now pools the shared feature across evaluators and
    confirms only planted preferences (9 of the 10 it can reach, no false findings).
- **Recommended configuration** (third run): revised proposer and ridge screen,
  `--twin ridge --twin-pairs 20 --workers 4 --proposer-prompt v2 --hypotheses 6 --examples 20`.
  The planted feature was proposed for the right evaluator for 15 of 16 preferences (the
  current prompt: 7), and **7 were confirmed, all correct**, with no null evaluator given a
  rule, for $10.90. Across three real-model runs, 14 of 14 confirmations are correct. The
  remaining losses are downstream of the proposer: five true hypotheses failed the
  word-count screen (sector and writing-style edits), three were tested but not significant
  at 30 re-ratings, and two battery slots were spent on a vague feature and on a feature
  with too few usable edits. See [`outputs/proposer_eval/`](outputs/proposer_eval/README.md)
  for the proposer test.
- **Second run, with the fixes** (`--twin ridge --twin-pairs 20 --workers 4`, same world,
  seed and budget): 80 hypotheses, 20 passed the screen and pooled into 10 features, 5
  batteries testing 13 evaluator-feature pairs, **6 confirmed, all 6 correct**, no null
  evaluator given a rule. Recovered planted preferences: the same-gender cue for Evaluators
  0, 11 and 16 (the men's "male founder" statements pooled onto the female-founder feature
  with the direction flipped), home-based for Evaluators 3 and 13, paying customers for
  Evaluator 15; estimates +0.39 to +0.55 against +0.50 planted. The seven rejections were
  also right. 62 minutes, $8.49. The ten planted preferences still missed were never
  proposed in a form that survived screening; with five batteries per run, recall is bounded
  by the proposer. Details in [`outputs/llm_sonnet/RUN_SUMMARY.md`](outputs/llm_sonnet/RUN_SUMMARY.md).

## What the benchmark does not show

- **The hypothesis space is closed.** All planted features are in the ten-operator edit
  library. The offline proposer chooses among ten candidates rather than generating new
  ones. Reaching features outside a fixed library is the LLM proposer's job, and the
  benchmark does not test it.
- **The model steps are stand-ins.** The benchmark shows that the loop's logic finds and
  confirms planted rules. It does not show that an LLM proposes and edits as cleanly.
  `llm-demo` on the same synthetic evaluators is the test of that.
- **Confirmation assumes the original evaluators re-rate.** The battery needs them to re-rate
  edited applications months after their first rating, each application at most once. With
  fresh evaluators it tests reactions at the panel level.
- **The evaluators are simple.** Their rules are additive and single-feature, and the style
  features suit the generator. The gender operator switches name and pronouns together, so
  the loop cannot tell which of the two triggers a rule.
- **The distributions are stylised.** Attribute distributions should be replaced by the real
  marginals before the synthetic set stands in for the study data.

## Layout

```
evloop/testbed.py      applications, renderer, detectors, edit and redaction operators
evloop/evaluators.py   evaluators with shared reactions and planted rules, outcomes, re-rating oracle
evloop/loop.py         the loop (observe, propose, screen, confirm, update), offline backend, baselines
evloop/llm.py          OpenAI-compatible client and the LLM backend
evloop/realdata.py     uploaded applications and ratings: loading, checks, screening, exports
evloop/present.py      plain-language summaries of runs for the app and the web explorer
evloop/app.py          the local app (Streamlit)
explorer/              the shareable web explorer and its pre-computed runs
scripts/               pre-computing the explorer's runs and building its data file
evloop/benchmark.py    replication grid and summary tables
evloop/report.py       figures and outputs/REPORT.md
docs/technical_note.md specification for the technical appendix
tests/                 unit tests and a stub LLM server
outputs/               benchmark rows, summaries, figures, report
outputs/llm_sonnet*/   the three language-model runs: audit trails, saved calls (llm_cache.tar.gz)
                       and call logs (llm_calls.jsonl.gz); RUN_SUMMARY.md describes all three
```

## License

MIT. See `LICENSE`.

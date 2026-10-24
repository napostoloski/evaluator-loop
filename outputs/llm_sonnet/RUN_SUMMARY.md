# Real-model run: Claude Sonnet 5.5, seed 1 (23 October 2026)

First end-to-end run of the LLM path on a real language model, on the same synthetic
world as the offline benchmark (156 applications, 20 evaluators, planted preferences of
0.5 points, re-rating budget 150 per evaluator, battery of 30 applications per feature).

```
python -m evloop llm-demo --base-url https://api.anthropic.com/v1 --model claude-sonnet-5-5 \
    --temperature -1 --seed 1 --extra-body '{"thinking": {"type": "between_tools"}}' --out outputs/llm_sonnet
```

The model proposed the hypotheses, rewrote the applications, judged whether a feature was
present in a text, and acted as the AI copy of each evaluator. The simulated evaluators
re-rated the edited texts, as in the benchmark. About 4,900 live model calls across the
attempts, 8.7M input and 1.1M output tokens, roughly $29 at list prices; the final run alone
is 4,290 live calls. The audit trail is `llm_demo_seed1_audit_trail.csv`; request and reply
logs are kept locally and not committed.

## Headline

| | |
|---|---|
| Hypotheses proposed (20 evaluators) | 80, four per evaluator, all in plain language |
| Passed the AI-copy screen | 27 |
| Tested in a re-rating battery | 5 |
| Confirmed | 1, and it was correct |
| False confirmations | 0 |
| Planted preferences recovered | 1 of 16 |

The one confirmed rule is Evaluator 0's planted same-gender cue: "the application is led by
a woman founder (female name and 'she' pronouns)", estimated at +0.45 against the panel
(planted +0.50), q = 0.0004. Where the right hypothesis reached the battery, the
confirmation step did exactly what it should.

The low recall is not a failure of the confirmation logic. It comes from three weaknesses
in the AI steps that the offline benchmark could not show, because its stand-ins do not
have them.

## What went wrong, and what to change

**1. Free-text hypotheses were not pooled across evaluators.** The battery stage ranks
features by how many evaluators request them. With a language model, the same idea arrives
in different words from each evaluator ("runs from the founder's home" five ways, "holds a
UK master's degree" three ways), so every hypothesis counted as its own feature with one
vote. The five batteries went to five near-arbitrary singletons, and 22 of the 27 passing
hypotheses were never tested, including correct ones for Evaluators 13 (home-based) and
15 (paying customers). Fix: one model call after screening that groups the passing
hypotheses into distinct features with a canonical wording, so votes and batteries pool.

**2. The language-model copy is noisier than the ridge copy.** It predicts from 10 edited
pairs per hypothesis, against 156 applications and 50 refits offline. Its intervals are
wide, so it failed true hypotheses (Evaluator 3's home-based, predicted +0.12) and passed
spurious ones (family business, co-founders, a UK degree, generic marketing language).
Fix: 30 pairs rather than 10, and rank by predicted effect rather than requiring the
interval to exclude zero at this sample size.

**3. The edits are not reliably minimal and complete.** Of 1,822 rewrites, 37 were caught
by the similarity and manipulation checks, but many that passed were half-edits. For the
gender edit the model sometimes switched pronouns and kept the male name, and once left
its own commentary in the text ("Wait, the name must..."). Measured against the
benchmark's pattern detectors, the rewrites switched the intended feature in only 14% to
56% of the cases that needed a change, depending on the feature; the exception was writing
fluency (90%). Part of this is the stand-in's rigidity (it recognises a fixed list of
names), but a name-and-pronoun mismatch is a bad edit by any reader's standard. Fix: a
stricter manipulation check that verifies the specific markers named in the hypothesis
(name and pronouns together), a reject-and-retry on any reply containing commentary, and a
hand audit of a sample before any human sees an edit.

**4. The proposer is plausible but often confounded.** For the four evaluators with a
planted same-gender cue, it proposed the cue itself only once; elsewhere it proposed
features that travel with gender in this world (textiles and handicrafts) or spurious
ones. That is the confusion the re-ratings exist to resolve, and it argues for showing
the proposer more contrasting examples and for pooling (fix 1) so that the cue hypothesis
of one evaluator gets tested on the others too.

Two prompt-level problems were found and fixed during the run, before these results:
Claude models reason before answering, which emptied the short replies until hidden
reasoning was turned off (`--extra-body`), and the copy explained even when asked for a
number only, which is handled by a tagged final line (`DEVIATION=+0.35`).

## What this means

- The loop's logic transfers: with a real model in every AI role, a correct hypothesis
  that reached the battery was confirmed with the right size and no false confirmations.
- The three fixes above are mechanical and cheap. A second run with them costs about the
  same ($25 to $30) and takes three to four hours.
- For Stage 2 the lesson is mainly about the edits: they need a verification step that
  checks the named markers, and a human audit, before any evaluator re-rates them.

## Fixes made after the run (same day, tested without model calls)

1. **Pooling.** `LLMBackend.canonicalize` runs once per round after screening: one model
   call groups the passing hypotheses into distinct features with a canonical statement
   and a polarity for each member. Equivalent statements share a feature key, so votes and
   batteries pool; a member stated as the negation of the canonical statement has its
   direction flipped; duplicates within one evaluator are collapsed. An offline fallback
   groups by word overlap if the model's reply is unusable.
2. **Screening.** Default 30 pairs per hypothesis (`--twin-pairs`). A pair is now the
   original plus one rewrite in the opposite direction, decided by the coder, so one
   rewrite per application instead of two, and a rewrite that comes back unchanged is
   dropped. `--twin ridge` screens the model's rewrites with the offline ridge copy,
   which needs no twin calls.
3. **Edits.** `edit_sanity` rejects rewrites whose section headings differ from the
   original, that introduce mixed pronouns, or that change pronouns without changing a
   name; `_clean_rewrite` strips appended commentary before the checks. Battery edits
   additionally go through the coder (as before) and a verifier call that compares
   original and rewrite for minimality and contradictions (`--no-verify-edits` to skip).
4. **Proposer.** The prompt now asks for one concrete, checkable property per
   hypothesis, not lists of alternatives.

Replay of the new edit checks over this run's 1,822 rewrites
(`python -m evloop audit-edits --log outputs/llm_sonnet/llm_calls.jsonl`), restricted to
rewrites where the hypothesis maps to a benchmark detector and a change was needed:

| feature | needed | old checks accepted / switched detector | new checks accepted / switched detector |
|---|---|---|---|
| gender cue | 81 | 79 / 40 (51%) | 40 / 40 (100%) |
| home-based | 70 | 70 / 70 (100%) | 70 / 70 (100%) |
| family business | 123 | 123 / 103 (84%) | 120 / 103 (86%) |
| foreign degree | 50 | 50 / 37 (74%) | 46 / 37 (80%) |
| solo founder | 91 | 91 / 46 (51%) | 67 / 46 (69%) |
| paying customers | 30 | 30 / 17 (57%) | 30 / 17 (57%) |
| writing fluency | 20 | 20 / 18 (90%) | 20 / 18 (90%) |
| all | 465 | 463 / 331 (71%) | 393 / 331 (84%) |

The new checks rejected 72 rewrites: 50 that came back unchanged, 21 that switched
pronouns but kept the name, 1 that contained only commentary. Every half-edit of the
gender cue is now caught. The remaining misses for solo founder and paying customers are
mostly rewrites faithful to a differently worded hypothesis (a survey count rather than
paying customers; a grammar fix rather than a team change) that the benchmark's narrow
detectors do not read, plus some genuinely weak edits that only the verifier call can
catch.

On the stub server, with the stub made to produce half-edits and appended commentary on
purpose, the full demo now pools the gender feature across 16 evaluators, flips the
polarity of the "man" statements, catches 61 half-edits, and confirms only planted
preferences (5 of 5 with the stub's few-shot copy, 9 of the 10 reachable ones with the
ridge copy, no false findings).

## Second run, with the fixes: `--twin ridge`, 20 pairs, 4 workers (same evening)

```
python -m evloop llm-demo --base-url https://api.anthropic.com/v1 --model claude-sonnet-5-5 \
    --temperature -1 --seed 1 --extra-body '{"thinking": {"type": "between_tools"}}' \
    --twin ridge --twin-pairs 20 --workers 4 --out outputs/llm_sonnet_ridge
```

Same world, same seed, same budget. The model proposed, rewrote, coded and verified; the
offline ridge copy screened the model's rewrites; the simulated evaluators re-rated.
4,219 live calls, 62 minutes, $8.49. Audit trail: `../llm_sonnet_ridge/llm_demo_seed1_audit_trail.csv`;
pooled groups: `../llm_sonnet_ridge/llm_demo_seed1_groups.json`.

| | first run | second run |
|---|---|---|
| Hypotheses proposed | 80 | 80 |
| Passed the screen | 27 | 20, forming 10 distinct features after pooling |
| Features given a battery | 5 (all singletons) | 5 (four requested by 2 to 4 evaluators each) |
| Evaluator-feature tests | 5 | 13 |
| Confirmed | 1 | 6 |
| Correct among confirmed | 1 | 6 |
| Null evaluators given a rule | 0 of 6 | 0 of 6 |
| Planted preferences recovered | 1 of 16 | 6 of 16 |
| Cost, time | $21, 3.5 h | $8.49, 62 min |

The six confirmed rules, with the planted value +0.50 in every case:

| evaluator | planted | confirmed statement | estimate | q |
|---|---|---|---|---|
| 0 (F) | same-gender cue | led by a female founder, direction + | +0.53 | 0.00005 |
| 3 (F) | home-based | runs from the founder's home, direction + | +0.55 | 0.003 |
| 11 (M) | same-gender cue | led by a female founder, direction − | +0.49 | 0.00007 |
| 13 (M) | penalises home-based | runs from the founder's home, direction − | +0.55 | 0.00001 |
| 15 (M) | paying customers | pilot with a specific number of paying customers | +0.39 | 0.008 |
| 16 (M) | same-gender cue | led by a female founder, direction − | +0.55 | 0.0004 |

The seven rejections were also right: Evaluator 0 and 11 on home-based, 13 on the gender
cue, 5 and 10 on survey counts, 6 on unit economics, and null Evaluator 17 on paying
customers (estimate −0.04, q = 0.84). Evaluator 3 is the case the walkthrough animation
describes: her home-based preference was proposed, pooled with three other evaluators'
wording, and confirmed at +0.55, while the gender cue was not attributed to her.

What the fixes did, visibly: pooling turned 20 differently worded hypotheses into 10
features and sent four shared ones to batteries, where 13 evaluator-feature pairs were
tested instead of 5; the polarity step mapped the men's "male founder" statements onto
the female-founder feature with the direction flipped (Evaluators 11 and 16); the edit
checks rejected 133 rewrites (88 pronoun switches without a name change, 30 by the
verifier, 13 unchanged, 2 by the manipulation check) before any re-rating was spent on
them.

What is still missed, and why: the ten remaining planted preferences were never proposed
in a form that survived screening. For Evaluators 1, 6 and 10 the proposer offered
sector or style features instead of the gender cue; for 2 and 12 the sector hypotheses
were worded as long contrasts and failed the screen; for 4 and 14 (writing fluency)
nothing close was proposed; for 5 and 6 (paying customers) the proposer offered survey
counts and unit economics, which are near misses. With five batteries per run and a
closed budget, recall is bounded by what the proposer puts forward; the offline
benchmark's proposer chooses from the ten library features and so cannot miss in this
way. The language-model proposer's open-endedness is the price of being able to find
features nobody listed, and the obvious next step is to let the pooled vote also admit
features that only one evaluator requested when budget remains, and to run more seeds.

Precision, which is what a Stage 2 human round depends on, was 6 of 6 here and 1 of 1 in
the first run.

## Third run: revised proposer (v2), ridge screen (24 October 2026, night)

```
python -m evloop llm-demo --base-url https://api.anthropic.com/v1 --model claude-sonnet-5-5 \
    --temperature -1 --seed 1 --extra-body '{"thinking": {"type": "between_tools"}}' \
    --twin ridge --twin-pairs 20 --workers 4 --proposer-prompt v2 --hypotheses 6 --examples 20 \
    --out outputs/llm_sonnet_v2
```

Same world, seed and budget. 5,300 live calls, $10.90 (the first-round proposals were reused
from the proposer test). The run was interrupted once by HTTP 529 (provider overloaded),
now retried by the client, and resumed from the cache.

| | run 2 (v1 proposer) | run 3 (v2 proposer) |
|---|---|---|
| hypotheses proposed | 80 | 120 |
| planted feature proposed for the right evaluator | 7/16 | 15/16 |
| confirmed | 6 | 7 |
| correct among confirmed | 6 | 7 |
| null evaluators given a rule | 0/6 | 0/6 |

New confirmation: Evaluator 1's gender cue (+0.59, q = 0.004). Across the three real-model
runs, 14 of 14 confirmations are correct.

Where the nine unconfirmed planted preferences were lost:
- proposed, failed the ridge screen (5): sector for Evaluators 2 and 12, writing fluency for
  4 and 14, paying customers for 5. The word-count copy underrates edits that change many
  words a little (sector) or only punctuation and capitalisation (fluency).
- tested, not significant (3): gender cue for Evaluator 6 (p = 0.05) and 10 (p = 0.09),
  paying customers for 6 (p = 0.20); the power limit of 30 re-ratings per feature.
- not proposed (1): paying customers for Evaluator 16.

Battery allocation also lost capacity: the vague feature "generic, unsupported claims"
won a battery on five requests (correctly rejected for all five), and the textile-
marketplace battery found only 4 usable edits. Candidate fixes, not yet made: skip a
battery that cannot reach a minimum size and give its slot to the next feature; rank
features by summed predicted effect rather than vote count; a screen that is less biased
against broad edits.

With 150 re-ratings per evaluator the offline benchmark recovers about 64% of half-point
preferences even with a proposer that never misses, so about 10 of 16 is the practical
ceiling on this seed; the real-model path now reaches 7, with no false findings.
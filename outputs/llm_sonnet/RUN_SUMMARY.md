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

A second real-model run with these fixes has not been made. Estimated cost: about $20 with
`--twin ridge`, $50 to $60 with the few-shot copy at 30 pairs.

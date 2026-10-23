# Improving the proposer (24 October 2026)

In the second real-model run every planted preference that was proposed for the right
evaluator was confirmed; the losses were all at the proposal stage. Two improvements were
tested.

## Step 1: revised prompt, real model, proposer only (3 seeds, $1.93)

`v1` is the current prompt (8 over- and 8 under-rated applications, 4 hypotheses). `v2`
shows 20 per side, asks for 6 distinct hypotheses, and has the model compare the groups on
the founder, the venture, the evidence and the writing before proposing. Scored by whether
each planted preference appears among the evaluator's hypotheses (keyword scorer, checked by
hand on seed 1).

| | v1 | v2 |
|---|---|---|
| planted feature proposed for the right evaluator | 26/48 (54%) | 41/48 (85%) |
| seeds 1 / 2 / 3 | 7 / 10 / 9 of 16 | 15 / 13 / 13 of 16 |
| gender cue | 11/18 | 18/18 |
| fluent English | 0/6 | 4/6 |
| paying customers | 4/12 | 7/12 |
| sector | 5/6 | 6/6 |
| home-based | 6/6 | 6/6 |

v2 changes three things at once, and more hypotheses per evaluator also means more
candidates downstream, so a full run is needed to see how much reaches confirmation and
whether precision holds. 44% of v2 hypotheses still contain an "or" despite the
instruction. Available as `--proposer-prompt v2 --hypotheses 6 --examples 20`.

## Step 2: generate, then measure (offline analogue, 12 seeds, free)

Proposer that ranks coded candidate features by a joint regression on the evaluator's
deviations (`OfflineBackend(proposer="coded")`), against the text lasso.

| proposer | recall | precision | null flagged | planted proposed | proposals per seed |
|---|---|---|---|---|---|
| text lasso | 0.635 | 0.904 | 0.069 | 0.917 | 80 |
| coded, p < 0.20 | 0.630 | 0.903 | 0.069 | 0.922 | 56 |
| coded, p < 0.35 | 0.641 | 0.918 | 0.069 | 0.938 | 70 |
| coded, p < 0.10 | 0.620 | 0.937 | 0.028 | 0.849 | 38 |

The statistical selection is as good as the tuned text model with fewer proposals. Offline
the proposer is not the bottleneck, so the benefit of step 2 can only show with the language
model; with v2 already at 85%, its remaining value is mainly for weak signals such as paying
customers.

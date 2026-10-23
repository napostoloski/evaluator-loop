# Who should be tested on a battery feature? (evaluated 24 October 2026)

After the second real-model run, every planted preference that was proposed for the right
evaluator was confirmed, and nine were lost because the feature was never proposed for that
evaluator, although seven of those features had been proposed for someone else. Since every
evaluator re-rates every battery anyway, two changes looked attractive: let the AI copy
nominate extra evaluators for pooled features, or test every evaluator on every battery
feature. Both are implemented as options (`LoopConfig.test_scope`, `prior_weight`) and were
compared with the default on the offline benchmark (12 seeds, delta 0.5, 150 re-ratings) and
on a cache replay of the real-model run (no model calls).

## Offline benchmark, 12 seeds

| rule | recall | precision | null evaluators flagged | true finds per seed vs default |
|---|---|---|---|---|
| proposers (default) | 0.635 | 0.904 | 0.069 | baseline |
| nominated by the copy | 0.609 | 0.929 | 0.069 | -0.4 (worse in 3, better in 1) |
| all evaluators, BH | 0.562 | 0.900 | 0.056 | -1.2 (worse in 7, better in 3) |
| all evaluators, weighted BH x4 | 0.599 | 0.891 | 0.042 | -0.6 |
| all evaluators, weighted BH x8 | 0.630 | 0.883 | 0.069 | -0.1 (worse in 3, better in 2) |
| nominated, then all, weighted x8 | 0.583 | 0.926 | 0.042 | -0.8 |
| all evaluators, heterogeneity adjustment | 0.312 | 0.984 | 0.000 | much worse |

## Real-model replay (Claude Sonnet 5.5, seed 1, from cache, no calls)

| rule | confirmed | correct |
|---|---|---|
| proposers (default) | 6 | 6 |
| all evaluators, BH | 6 | 6 (gains Evaluator 1, loses 15: q = 0.050) |
| all evaluators, weighted BH x8 | 7 | 7 |

## Conclusion

Neither change improves the loop on the benchmark. Copy nomination re-finds what the proposer
found (the copy is fitted to the same deviations) and adds about three tests per seed. Testing
everyone multiplies the tests by four, and the stricter false-discovery bar costs more true
findings than the wider coverage gains; weighted BH recovers the loss and is neutral. The
default stays. "All evaluators, weighted x8" is kept as an option: neutral offline, +1 on one
real-model seed. The bottleneck with a language-model proposer is proposing the right feature
for the right evaluator, which a proposer-only experiment can measure for a few dollars.

Files: `offline_batch1.csv`, `offline_batch2.csv` (per evaluator and variant), the two scripts,
and the audit trails of the two replays.

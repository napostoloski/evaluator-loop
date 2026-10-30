# Technical note: the evaluator loop and the planted-preference benchmark

This note specifies the prototype in enough detail to evaluate it and to serve as the basis of
the technical appendix. Code references are to the `evloop` package.

**Terms.** The proposal uses plain-language names for what the code and this note call by
shorter ones. A *preference* is a rule. The *statistical copy* of an evaluator is the twin.
The *LASSO* is the offline proposer in step 2. The *re-ratings* form the battery in step 4.
Table 1 of the proposal reports the methods called Full loop, Screen only, Twin only and
Library regression in Section 3 below; the README maps every number in the proposal to the
file it comes from.

## 1. Problem

Twenty evaluators each score the same 156 applications. Evaluator *j*'s scores mix three
things: what every evaluator sees in an application, the reactions every evaluator shares
(for example to reported paying customers), and what is particular to *j*.

The loop's job is to find the features of the application text that drive the particular
part. It returns them as a list of rules of the form "evaluator *j* scores applications
higher when feature *f* takes value *v*, by about *e* points more than the rest of the panel
does".

Each rule must be supported by a counterfactual test. Correlation with *j*'s scores is not
enough, and a reaction the whole panel shares is not a rule about *j*.

## 2. The loop (`loop.run_panel`)

The loop runs over the panel in rounds (default two; within the default budget the offline
benchmark completes in one).

**Step 1. Observe.** Let *c_i* be the mean score that the other 19 evaluators gave application
*i* (the leave-one-out consensus). Subtract any rules already confirmed for *j* from *j*'s
scores, then regress the result on *c* by OLS. The residual *r_ij* is *j*'s own component,
net of leniency, scale and the reactions the panel shares.

**Step 2. Propose (offline backend).** Each application is represented by three blocks of
columns, all standardised on the original applications:

- word and bigram TF-IDF (minimum document frequency 3, at most 600 terms, sublinear term frequency);
- 20 latent topics from LSA on the TF-IDF;
- five style statistics: the share of sentences starting in lower case, the share of sections
  without terminal punctuation, subject-verb agreement errors, the count of numbers, and
  length.

The style statistics were designed alongside the synthetic generator, so they suit its
writing slips. With real applications, an LLM proposer replaces this block.

The lasso penalty is chosen by five-fold cross-validation over 12 values, from 0.89 to 0.03
times the null penalty. The lasso is then refitted 25 times on random 75% subsamples at 0.6
times the cross-validated penalty. The scaling favours recall, because later steps remove
false proposals. Features selected in at least 40% of refits keep their mean coefficient,
giving a sparse weight vector *w*.

Each edit operator *k* has a lexical footprint *Δ_k*: the mean change in the representation
when the feature is switched from 0 to 1 in every application. The footprint is computed by
actually editing the texts.

A hypothesis is an operator with score *s_k = w·Δ_k*, with direction given by the sign of
*s_k*. Up to four are proposed per evaluator and round. Features already tested for *j*, or
rejected by *j*'s twin, are not proposed again for *j*.

**Step 3. Screen on a twin (offline backend).** The twin is a ridge regression of *r_j* on the
same representation. Its penalty is chosen by efficient leave-one-out cross-validation over
20 values from 10^-2 to 10^3, and it is refitted on 50 bootstrap resamples.

For each hypothesis (*k*, *v*), every application is edited to *f_k = v* and to
*f_k = 1 - v*, and the twin's predicted difference is averaged. A hypothesis passes if the
2.5% bootstrap quantile exceeds zero and the mean is at least 0.05 points.

The ridge twin is a second, dense model of the same residual. It is a cheap filter, not
independent evidence. An LLM twin (`llm.LLMTwin`) would be closer to independent.

**Step 4. Confirm with a re-rating battery.**

*Choosing features.* Features that passed some evaluator's screen in this round are ranked
by how many evaluators they passed for, then by the largest twin effect. Up to five enter
the battery, subject to the remaining budget and to the applications not yet used. A feature
batteried in an earlier round is tested again from the stored battery rather than re-run.

*Editing.* For each feature, 30 applications are drawn, half with the feature on and half
with it off. Each is edited so that the feature switches: off where it was on, on where it
was off. No application enters two batteries, so no evaluator re-rates an application twice
in a run.

*Re-rating.* Every evaluator re-rates every edited application once. The outcome is the
change *d_ij* = (re-rating of the edited version) − (original rating). This differences out
the application's quality, and part of the evaluator's own reading of it. The rest of that
reading is fresh noise on the second occasion.

*Estimation.* For each evaluator separately, *d* is regressed on an intercept (drift between
the two occasions) and the edit direction (+1 or −1). The slope is the evaluator's reaction
*γ_j*, with its own residual variance and about 28 degrees of freedom.

*The idiosyncratic test.* The test statistic is *γ_j* minus the median of the other
evaluators' reactions. The median keeps a minority with their own rule on the same feature
from moving the reference. Its variance is approximated as π/2 times the variance of the
mean.

*Multiplicity.* Benjamini-Hochberg q-values are computed across every test in the round,
panel-wide. A rule is accepted if *q* < 0.05 and the estimated effect is at least 0.10
points.

*Budget.* The default is 150 re-ratings per evaluator: five features of 30 applications.
Every evaluator re-rates the whole battery, so the cost is the same for all.

*Two optional adjustments* (`LoopConfig`):

- `het_adjust` adds the spread of reactions across evaluators, estimated robustly from the
  others (MAD), to the standard error, so that a rule must stand out from ordinary
  differences between evaluators.
- `rtm_adjust` adds the evaluator's original residual as a covariate. A hypothesis is chosen
  because the evaluator's original scores leaned one way on the feature. Part of that lean is
  noise that a re-rating does not repeat, so the change *d* regresses toward the mean and can
  mimic a reaction. The covariate removes this under the null. Because the original residual
  also contains the rule itself, it attenuates true effects.

Both are reported in the benchmark as variants. Both are off by default because each costs
much of the loop's power under realistic re-rating noise (see `outputs/REPORT.md`).

**Update.** Each accepted rule is subtracted from *j*'s scores, using the effect times
whether the rule fires on each original application as read by the coder. The residual is
then recomputed from scratch (step 1).

**Outputs.** For each evaluator the loop returns:

- the accepted rules;
- an audit trail of every hypothesis, with its proposer score and the terms behind it, its
  twin effect and interval, its battery estimate, standard error, p and q;
- the observed same-gender premium, and the premium that the accepted rules alone reproduce.

**Outcome test (`loop.outcome_label`).** For each accepted rule, the outcome is regressed on
the leave-one-out consensus and the coded indicator that the rule fires. A positive,
significant coefficient labels the rule "ability": the rewarded feature carries information
about performance that the consensus misses. Otherwise it is labelled "bias".

**Feature coder.** Updates, the outcome test and the regression baselines read features
through a coder that flips each coded value with probability 0.05. The synthetic evaluators
read the text exactly. In the real study the coder is an LLM or a human.

## 3. Methods compared (`benchmark.py`)

| Method | What it uses | What it can find |
|---|---|---|
| Screen only | Step 2; accepts proposals with score of at least 0.10 (one round) | Correlations in the text |
| Twin only | Steps 2 and 3; accepts what passes the screen and subtracts the shrunken twin effect (two rounds) | What a model of the evaluator believes |
| Full loop | Steps 2 to 4 | Rules confirmed as idiosyncratic by re-ratings |
| Loop, total-reaction test | As the full loop, but tests *γ_j* against zero | Shows why the panel contrast matters |
| Edit everything | No proposer or twin. The same budget is split over all ten library features (15 applications each at a budget of 150), every evaluator is tested on every feature in both directions, and BH runs over all tests (`loop.edit_everything`) | The natural design when the candidate list is short and known |
| Coded covariates | OLS of *r_j* on coded founder gender, city tier, sector cluster and solo founder; HC1 errors; BH across all evaluator-feature tests, as for the loop; effect of at least 0.10 | Rules on the usual structured variables |
| Library regression | The same on all ten library features as coded | The strongest method without edits |
| Redaction battery | Each evaluator re-rates 150 applications, a random half with gender cues redacted and half unchanged; the change is regressed on redaction, same-gender founder and their interaction | Only whether a gender premium runs through the cue |

## 4. The synthetic world

**Applications (`testbed.py`).** There are 156 applications with four sections (business
idea, problem to solve, target customer, team), rendered from attributes by templates. The
attributes carry built-in confounds:

- Founder gender shifts the sector cluster: 65% of women founders and 30% of men are in
  consumer sectors.
- Gender and consumer sector shift the chance of a home-based business, from about 14%
  (men, technology) to about 65% (women, consumer).
- City tier shifts fluency and the chance of a foreign degree.
- Latent quality raises stage and the chance of paying customers, and controls how specific
  the text is.

Every wording choice comes from a per-application stream keyed to fixed sentence slots.
Changing one attribute and re-rendering therefore changes only the sentences that express
it, and no other detected feature (`tests/test_testbed.py`).

**Edit operators.** Ten binary features:

- gender cue (name and pronouns, switched together);
- consumer sector;
- metropolitan city;
- fluent English;
- paying pilot customers;
- family business;
- foreign degree;
- solo founder;
- home-based;
- five or more years of experience.

There are also two redaction operators, for gender and geography.

**Evaluators (`evaluators.py`).** There are 20 evaluators, 10 women and 10 men. Evaluator
*j*'s latent score for an application is

3 + leniency_j + skill_j (quality + perception_ij) + shared reactions + planted rules.

The parameters are drawn as follows:

- leniency ~ N(0, 0.3), skill ~ U(0.35, 0.7);
- perception noise, the evaluator's reading of the venture, with SD ~ U(1.0, 1.6) per evaluator;
- shared reactions: paying customers +0.30, fluent English +0.15, experience +0.15, solo
  founder −0.15, each varying across evaluators with SD 0.10.

Five dimension scores are latent + N(0, 0.35), rounded and clipped to 1-5, and the overall
score is their mean. The mean correlation between two evaluators' scores is about 0.4.

On a re-rating, the evaluator's reading of the venture is ρ times the first reading plus
fresh noise (default ρ = 0.6), with fresh test-retest noise N(0, 0.15) and fresh dimension
noise. The test-retest correlation of an evaluator's ratings is then about 0.7. The
benchmark also reports ρ = 0.3 and 0.9.

Planted roles per gender:

- two **cue** evaluators: same-gender premium triggered by the founder's name and pronouns;
- one **sector proxy**: favours their own gender's typical sector cluster;
- one **home proxy**: women favour home-based ventures, men penalise them;
- one **fluency** evaluator: an extra premium on top of the shared one;
- one **ability** evaluator: an extra premium for paying customers;
- one **multi** evaluator: cue plus ability;
- three **null** evaluators.

**Outcomes.** log revenue = 0.8 quality + 0.5 paying customers + 0.2 stage + N(0, 1.5). The
consensus score correlates about 0.55 with the outcome.

## 5. Metrics

- **Precision and recall** of accepted (feature, value) rules against planted rules, pooled,
  with standard errors across replications. Shared reactions reported as an evaluator's own
  rule count as false rules.
- **Share of evaluators exactly right**: no missed rule and no false rule.
- **Null evaluators given a rule.**
- **Root accuracy.** This covers evaluators with a significant positive same-gender premium
  whose planted root is a cue or gender-correlated content. It is the share classified
  correctly as cue versus content. A further metric checks whether the method names the
  right content feature.
- **Effect estimates** for correctly recovered rules.
- **Ability or bias** labelling accuracy.

Benchmark seeds are 100 to 111. Development and the tests used seeds below 20.

## 6. What the benchmark does not show

- **The hypothesis space is closed.** All planted features are among the ten operators, so
  the offline proposer chooses among ten candidates and does not generate hypotheses. The
  LLM proposer is meant to reach features outside a fixed library. The benchmark does not
  test that.
- **The style block favours the generator**, as noted in step 2.
- **Name and pronoun are not separated.** The gender operator switches both together, so a
  confirmed cue rule means "name and pronouns", not "name".
- **Edits are exact here.** LLM rewrites will be noisier. `llm.edit_llm` checks word-level
  similarity, and `LLMBackend` drops edits that fail a manipulation check by the coder.
- **Confirmation assumes the original evaluators can re-rate.** The battery needs them to
  re-rate edited applications months after their first rating, each application once. With
  fresh evaluators the same battery tests reactions at the panel level, not for one
  individual.
- **Confirming individual rules is expensive.** Under realistic re-rating noise, an
  individual evaluator's rule of half a point needs about 30 or more re-ratings per tested
  feature. Screening is what makes that affordable within one pass through 156 applications.
  Smaller rules can be confirmed only for groups of evaluators.
- **Regression to the mean.** The default test does not remove the regression-to-the-mean
  bias that hypothesis selection creates; its effect shows up in the benchmark's precision.
  The covariate variant removes it at a large cost in power.
- **The outcome test has low power.** With 156 startups and noisy outcomes, it often fails
  to label an informative rule as ability.

## 7. From prototype to the real study

| Component | Prototype | Real study |
|---|---|---|
| Proposer | Lasso on text features, mapped to a library of edit operators | LLM reads the applications each evaluator over- and under-rates and states hypotheses (`llm.propose_llm`) |
| Edits | Exact re-rendering from attributes | LLM minimal rewrites with similarity and manipulation checks; a sample audited by a person |
| Feature coding | Detectors with 5% error | LLM coding (`llm.code_llm`), validated against human coding |
| Twin | Ridge text model of the residual | The same ridge copy, screening the LLM's edits (the configuration of the second and third real-model runs); a few-shot LLM copy (`llm.LLMTwin`) remains an option |
| Real data | Not applicable | `realdata.py` and the app load applications and ratings, run the loop up to screening, and export the edited applications for re-rating (`hypotheses.csv`, `edits_to_rerate.csv`) |
| Confirmation | Re-rating by synthetic evaluators | Stage 2 re-rating battery with the original evaluators: hypotheses pooled across the panel, up to five features of 30 edited applications each, every evaluator re-rating all of them, plus about 15 unedited applications to measure re-rating consistency |
| Outcomes | Simulated | Realized revenue and survival from December 2026 |

`python -m evloop llm-demo` runs the same loop with an LLM backend on the synthetic
evaluators. Those evaluators read edits through regular-expression detectors that accept
some common paraphrases but not all. The demo therefore measures partly whether the LLM's
edits change the features as these detectors perceive them. The LLM backend's
manipulation check uses the LLM coder instead.

## 8. Real-model runs

On 23 October 2026 the LLM path ran end to end on Claude Sonnet 5.5 (seed 1, same world
and budget as the benchmark). One of 16 planted preferences was confirmed, correctly and
with the right size (Evaluator 0's gender cue, +0.45 against +0.50 planted, q = 0.0004),
with no false confirmations. Recall was low for three reasons that the offline stand-ins
cannot exhibit: hypotheses phrased differently by different evaluators were not pooled,
so the battery budget went to singletons; the few-shot copy predicts from 10 pairs and is
noisy; and many rewrites were incomplete (pronouns switched, name kept) yet passed the
checks. `outputs/llm_sonnet/RUN_SUMMARY.md` gives the numbers. The fixes are implemented: a
canonicalisation call after screening that merges equivalent hypotheses and aligns their
polarity before the batteries; 30 pairs for the copy, with a ridge-copy option that needs
no model calls; offline sanity checks on every rewrite (sections, pronoun consistency,
unchanged text, commentary) and a verifier call on battery edits. Replayed over the first
run's rewrites, the checks raise the share of accepted edits that switch the intended
feature from 71% to 84%, and to 100% for the gender cue.

A second run the same evening with the fixes (ridge screen on the model's rewrites, 20
pairs, same world and budget) confirmed 6 of 16 planted preferences, all correct, with no
false confirmations and seven correct rejections, for $8.49 in 62 minutes: the gender cue
for three evaluators (two of them via pooled, polarity-flipped "male founder" statements),
home-based for two, paying customers for one. The remaining misses were never proposed in
a form that survived screening, so with five batteries per run recall is bounded by the
language-model proposer. Precision across both runs is 7 of 7.

A third run with a revised proposer prompt (20 contrasting applications per side, six
distinct hypotheses, an explicit comparison of founder, venture, evidence and writing)
proposed the planted feature for the right evaluator for 15 of 16 preferences, against 7
with the original prompt, and confirmed 7 of 16, all correct ($10.90). A proposer-only test
on three seeds gave 85% against 54%. With the proposer no longer the bottleneck, the
remaining losses are the word-count screen's bias against broad edits (sector, writing
style), the power of 30 re-ratings per feature, and battery allocation by vote count.
Across the three real-model runs, 14 of 14 confirmations are correct and no evaluator
without a planted preference was given a rule.

**Records and replay.** Every model call is cached with its full request and reply. The
caches of all three runs are in `outputs/llm_sonnet*/llm_cache.tar.gz` and the call logs in
`llm_calls.jsonl.gz` beside them. The third run replays exactly from its cache with the
current code: no model calls are made and the same seven preferences are confirmed with the
same estimates (README, "Reproducing the results"). The first two runs used earlier versions
of the code, so their caches serve as an audit record rather than for an exact replay.

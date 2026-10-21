"""Planted-rule benchmark: does the loop find what we planted, and nothing else?

Each replication draws new applications and a new panel, plants rules of size `delta`, and
runs every method on the same applications:

    screen      proposer only, no edit tests (a purely correlational reading of the text)
    twin        proposer + twin screen, no real queries
    loop        proposer + twin screen + re-rating battery, testing each evaluator's reaction
                relative to the panel (the full loop)
    loop_total  ablation: the same battery, but testing each evaluator's total reaction
    edit_all    no proposer or twin: the same re-rating budget spread over all ten library
                features, every evaluator tested on every feature
    coded       regression of the residual on the usual structured covariates
                (founder gender, city tier, sector cluster, solo founder); no edits
    library     the same regression on all ten features in the edit library, as coded by a
                coder with a 5% error rate: the strongest no-edit baseline
    redaction   prereg-style diagnostic for the gender premium only: a re-rating battery of
                gender-redacted applications with a budget at least as large as the loop's

Benchmark seeds start at 100 so that they do not overlap the seeds used during development
and in the tests.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .evaluators import make_world
from .loop import (CODED, LIBRARY, LoopConfig, TextSpace, edit_everything, outcome_label, redaction_battery,
                   regression_panel, run_panel, trail_records)
from .testbed import sample_applications

PANEL_METHODS = {"screen": "none", "twin": "twin", "loop": "panel", "loop_total": "total"}


def PROXY_KEYS(own):
    return {("consumer_sector", own), ("home_based", own)}


def _true_root(role):
    return {"cue": "cue", "multi": "cue", "proxy": "content", "proxy_home": "content"}.get(role, "none")


def _root_from_rules(rules, own):
    keys = {(f, t) for f, t, *_ in rules}
    if ("gender_cue", own) in keys:
        return "cue"
    if keys:
        return "content"
    return "unexplained"


VARIANTS = {"loop_het": dict(het_adjust=True), "loop_rtm": dict(rtm_adjust=True)}


def run_job(seed: int, delta: float, battery_apps: int,
            methods=("screen", "twin", "loop", "loop_total", "edit_all", "coded", "library", "redaction"),
            rho: float = 0.6, with_trails: bool = False):
    t0 = time.time()
    apps = sample_applications(156, seed=seed)
    space = TextSpace(apps, seed=seed)
    n_features = min(5, 150 // battery_apps)
    budget = n_features * battery_apps
    rows, trails = [], []
    for method in methods:
        world = make_world(apps, seed=seed, delta=delta, rho=rho)
        if method in PANEL_METHODS or method in VARIANTS:
            cfg = LoopConfig(confirm=PANEL_METHODS.get(method, "panel"), seed=seed, battery_apps=battery_apps,
                             battery_features=n_features, budget=budget, **VARIANTS.get(method, {}))
            results = run_panel(world, space, cfg)
        elif method == "redaction":
            red = redaction_battery(world, np.random.default_rng(seed * 7919), n_apps=min(budget, len(apps)))
        elif method == "edit_all":
            everything = edit_everything(world, budget, np.random.default_rng(seed * 104729))
        elif method in ("coded", "library"):
            regs = regression_panel(world, space, CODED if method == "coded" else LIBRARY)
        for j, ev in enumerate(world.panel):
            own = 1 if ev.gender == "F" else 0
            planted = {r.key for r in ev.rules}
            row = dict(seed=seed, delta=delta, rho=rho, battery_apps=battery_apps, budget=budget, method=method,
                       eid=j, gender=ev.gender, role=ev.role, true_root=_true_root(ev.role),
                       planted=json.dumps(sorted(planted)))
            if method in PANEL_METHODS or method in VARIANTS:
                res = results[j]
                acc = res.accepted
                row.update(queries=res.queries, observed_premium=res.observed_premium,
                           observed_premium_p=res.observed_premium_p, rule_twin_premium=res.twin_premium,
                           root=_root_from_rules(acc, own),
                           rounds=json.dumps([h.round for h in res.trail if h.accepted]))
                if with_trails:
                    for rec in trail_records(res):
                        rec.update(seed=seed, delta=delta, battery_apps=battery_apps, method=method)
                        trails.append(rec)
                if method == "loop":
                    row["outcome_labels"] = json.dumps({f"{f}:{t}": outcome_label(world, space, j, f, t)
                                                        for f, t, _ in acc})
            elif method in ("coded", "library"):
                acc = regs[j]
                row.update(queries=0, root=_root_from_rules(acc, own))
            elif method == "edit_all":
                acc = everything[j]
                row.update(queries=world.queries_used.get(j, 0), root=_root_from_rules(acc, own))
            else:
                acc = []
                row.update(queries=world.queries_used.get(j, 0), root=red[j][0], redaction_p=red[j][1])
            got = {(f, t) for f, t, *_ in acc}
            row.update(accepted=json.dumps(sorted(got)),
                       effects=json.dumps({f"{f}:{t}": round(e, 3) for f, t, e in acc}),
                       tp=len(got & planted), fp=len(got - planted), fn=len(planted - got))
            if method == "redaction":
                row.update(tp=np.nan, fp=np.nan, fn=np.nan)
            rows.append(row)
    return rows, trails, round(time.time() - t0, 1)


def _job(args):
    return run_job(*args)


def run_grid(out_dir: str, seeds=range(100, 112), deltas=(0.25, 0.5, 0.75), main_apps: int = 30,
             sweep_apps=(15, 50), sweep_rhos=(0.3, 0.9), sweep_delta: float = 0.5, workers: int = 2, log=print):
    """Main grid over effect sizes, plus (at the middle effect size) a sweep over applications per
    feature, a sweep over the test-retest correlation rho, and two variants of the confirmation test."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, d, main_apps) for s, d in itertools.product(seeds, deltas)]
    jobs += [(s, sweep_delta, b, ("loop", "edit_all")) for s, b in itertools.product(seeds, sweep_apps)]
    jobs += [(s, sweep_delta, main_apps, ("loop", "edit_all", "library"), r)
             for s, r in itertools.product(seeds, sweep_rhos)]
    jobs += [(s, sweep_delta, main_apps, tuple(VARIANTS)) for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (r, _, secs) in enumerate(ex.map(_job, jobs)):
            rows.extend(r)
            log(f"[{k + 1}/{len(jobs)}] {jobs[k]} ({secs}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "benchmark_rows.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def recovery_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled precision and recall, plus the standard error across replications."""
    d = df[df.method != "redaction"].copy()
    d["exact"] = (d.fp == 0) & (d.fn == 0)
    d["null_fp"] = (d.role == "null") & (d.fp > 0)
    keys = ["method", "delta", "budget", "battery_apps", "rho"]
    g = d.groupby(keys)
    per_seed = d.groupby(keys + ["seed"]).agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
    per_seed["precision"] = per_seed.tp / (per_seed.tp + per_seed.fp).replace(0, np.nan)
    per_seed["recall"] = per_seed.tp / (per_seed.tp + per_seed.fn)
    se = per_seed.groupby(keys).agg(precision_se=("precision", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
                                    recall_se=("recall", lambda x: x.std(ddof=1) / np.sqrt(x.count())))
    out = pd.DataFrame({
        "precision": g.tp.sum() / (g.tp.sum() + g.fp.sum()),
        "recall": g.tp.sum() / (g.tp.sum() + g.fn.sum()),
        "exact_share": g.exact.mean(),
        "null_false_positive_rate": d[d.role == "null"].groupby(keys).null_fp.mean(),
        "false_rules_per_evaluator": g.fp.mean(),
        "queries_per_evaluator": g.queries.mean(),
        "replications": g.seed.nunique(),
    }).join(se)
    return out.reset_index()


def recall_by_role(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.method != "redaction") & (df.role != "null")]
    g = d.groupby(["method", "delta", "budget", "battery_apps", "rho", "role"])
    return (g.tp.sum() / (g.tp.sum() + g.fn.sum())).rename("recall").reset_index()


def root_table(df: pd.DataFrame, budget: int = 150, rho: float = 0.6) -> pd.DataFrame:
    """Among evaluators who show an apparent same-gender premium, is its root identified?"""
    df = df[df.rho == rho]
    loop = df[(df.method == "loop") & (df.budget == budget) & (df.battery_apps == 30)][
        ["seed", "delta", "eid", "observed_premium", "observed_premium_p"]]
    flag = loop[(loop.observed_premium > 0) & (loop.observed_premium_p < 0.05)]
    keys = flag[["seed", "delta", "eid"]]
    d = df[df.method.isin(["loop", "loop_total", "edit_all", "twin", "library", "coded", "redaction"]) & (df.budget == budget)
           & (df.battery_apps == 30)].merge(keys, on=["seed", "delta", "eid"])
    d = d[d.true_root.isin(["cue", "content"])].copy()

    def correct(r):
        if r.method == "redaction":
            return (r.true_root == "cue") == (r.root == "cue")
        return r.root == r.true_root

    def specific(r):
        if r.method == "redaction":
            return np.nan
        acc = {tuple(x) for x in json.loads(r.accepted)}
        own = 1 if r.gender == "F" else 0
        planted = {tuple(x) for x in json.loads(r.planted)}
        if r.true_root == "cue":
            return float(("gender_cue", own) in acc and not (acc & PROXY_KEYS(own)))
        content = {k for k in planted if k[0] != "gender_cue"}
        return float(content <= acc and ("gender_cue", own) not in acc)

    d["correct_root"] = d.apply(correct, axis=1)
    d["correct_specific"] = d.apply(specific, axis=1)
    g = d.groupby(["method", "delta"])
    return pd.DataFrame({"evaluators_with_apparent_premium": g.size(),
                         "root_accuracy": g.correct_root.mean(),
                         "names_the_right_feature": g.correct_specific.mean()}).reset_index()


def effect_table(df: pd.DataFrame, budget: int = 150, rho: float = 0.6) -> pd.DataFrame:
    d = df[(df.method.isin(["loop", "loop_total", "edit_all", "twin", "screen", "library"])) & (df.budget == budget)
           & (df.rho == rho) & (df.battery_apps == 30)]
    recs = []
    for _, r in d.iterrows():
        planted = {tuple(x) for x in json.loads(r.planted)}
        for k, e in json.loads(r.effects).items():
            f, t = k.rsplit(":", 1)
            if (f, int(t)) in planted:
                recs.append(dict(method=r.method, delta=r.delta, estimate=e))
    e = pd.DataFrame(recs)
    g = e.groupby(["method", "delta"]).estimate
    return pd.DataFrame({"mean_estimate": g.mean(), "sd_estimate": g.std(), "n": g.size()}).reset_index()


def ability_table(df: pd.DataFrame, budget: int = 150, rho: float = 0.6) -> pd.DataFrame:
    d = df[(df.method == "loop") & (df.budget == budget) & (df.rho == rho) & (df.battery_apps == 30)]
    recs = []
    for _, r in d.iterrows():
        planted = {tuple(x) for x in json.loads(r.planted)}
        for k, lab in json.loads(r.outcome_labels).items():
            f, t = k.rsplit(":", 1)
            if (f, int(t)) in planted:
                truth = "ability" if f == "traction" else "bias"
                recs.append(dict(delta=r.delta, feature=f, truth=truth, label=lab, correct=lab == truth))
    a = pd.DataFrame(recs)
    a["labelled_ability"] = a.label == "ability"
    return a.groupby(["delta", "truth"]).labelled_ability.agg(["mean", "size"]).reset_index().rename(
        columns={"mean": "share_labelled_ability", "size": "n_rules"})

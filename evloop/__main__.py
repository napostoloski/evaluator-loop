"""Command line.

    python -m evloop demo --seed 1 --delta 0.5
    python -m evloop benchmark --seeds 12 --workers 2   # seeds 100-111
    python -m evloop report
    python -m evloop llm-demo --base-url http://localhost:8000/v1 --model Qwen/Qwen3.8-27B --no-thinking
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def _design(apps_per_feature: int):
    """Features per battery round and total budget for a given number of applications per feature."""
    k = max(1, min(5, 150 // apps_per_feature))
    return k, k * apps_per_feature


def cmd_demo(a):
    from .evaluators import make_world
    from .loop import LoopConfig, TextSpace, run_panel, trail_records
    from .testbed import sample_applications

    apps = sample_applications(156, seed=a.seed)
    world = make_world(apps, seed=a.seed, delta=a.delta)
    space = TextSpace(apps, seed=a.seed)
    k, budget = _design(a.battery_apps)
    cfg = LoopConfig(seed=a.seed, battery_apps=a.battery_apps, battery_features=k, budget=budget)
    results = run_panel(world, space, cfg)  # world uses the default test-retest correlation (rho = 0.6)
    rows, trail = [], []
    print(f"{'eval':>4} {'g':>1} {'planted role':<11} {'planted rules':<34} {'accepted rules (estimated effect)':<46} re-ratings")
    for j, ev in enumerate(world.panel):
        res = results[j]
        planted = ", ".join(f"{r.feature}={r.target}" for r in ev.rules) or "-"
        got = ", ".join(f"{f}={t} ({e:+.2f})" for f, t, e in res.accepted) or "-"
        print(f"{j:>4} {ev.gender:>1} {ev.role:<11} {planted:<34} {got:<46} {res.queries}")
        rows.append(dict(eid=j, gender=ev.gender, role=ev.role, planted=planted, accepted=got,
                         queries=res.queries, observed_premium=res.observed_premium,
                         rule_twin_premium=res.twin_premium))
        trail.extend(trail_records(res))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"demo_seed{a.seed}_summary.csv"), index=False)
    pd.DataFrame(trail).to_csv(os.path.join(a.out, f"demo_seed{a.seed}_audit_trail.csv"), index=False)
    print(f"\nwrote {a.out}/demo_seed{a.seed}_summary.csv and the full audit trail")


def cmd_benchmark(a):
    from .benchmark import run_grid
    from .report import build_report

    run_grid(a.out, seeds=range(100, 100 + a.seeds), deltas=tuple(a.deltas), main_apps=a.battery_apps,
             sweep_apps=tuple(a.sweep_apps), workers=a.workers)
    print("report:", build_report(os.path.join(a.out, "benchmark_rows.csv"), a.out, _design(a.battery_apps)[1]))


def cmd_report(a):
    from .report import build_report
    print(build_report(os.path.join(a.out, "benchmark_rows.csv"), a.out, _design(a.battery_apps)[1]))


def cmd_llm_demo(a):
    import numpy as np

    from .evaluators import make_world
    from .llm import LLMBackend, LLMClient
    from .loop import LoopConfig, run_panel, trail_records
    from .testbed import OPERATORS, detect, sample_applications

    apps = sample_applications(156, seed=a.seed)
    world = make_world(apps, seed=a.seed, delta=a.delta, rho=a.rho)
    extra = {}
    if a.no_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if a.top_p is not None:
        extra["top_p"] = a.top_p
    client = LLMClient(a.base_url, a.model, api_key=a.api_key, seed=a.seed, extra_body=extra,
                       temperature=None if a.temperature < 0 else a.temperature,
                       max_tokens_field=a.max_tokens_field,
                       cache_dir=os.path.join(a.out, "llm_cache"), log_path=os.path.join(a.out, "llm_calls.jsonl"))
    backend = LLMBackend(client, world)
    k, budget = _design(a.battery_apps)
    cfg = LoopConfig(seed=a.seed, battery_apps=a.battery_apps, battery_features=k, budget=budget, max_rounds=a.rounds)
    results = run_panel(world, backend, cfg)
    texts = [x.text for x in apps]
    truth = {op.name: np.array([detect(op.name, t) == 1 for t in texts], dtype=float) for op in OPERATORS}
    rows = []
    for j, res in results.items():
        ev = world.panel[j]
        print(f"\nevaluator {j} ({ev.gender}, planted {[(r.feature, r.target) for r in ev.rules]})")
        for h in res.trail:
            if not h.accepted:
                continue
            f = backend.fires(h.feature, h.description, 1)
            # which library feature does the LLM's hypothesis correspond to? (evaluation only)
            best, phi = None, 0.0
            for name, t in truth.items():
                if f.std() > 0 and t.std() > 0:
                    c = abs(np.corrcoef(f, t)[0, 1])
                    if c > phi:
                        best, phi = name, c
            print(f"  ACCEPTED {h.description} (direction {h.target}, effect {h.human_effect:+.2f}); "
                  f"closest library feature: {best} (|r| = {phi:.2f})")
        rows.extend(trail_records(res))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"llm_demo_seed{a.seed}_audit_trail.csv"), index=False)
    print(f"\nedits dropped by the similarity or manipulation check: {backend.dropped_edits}")


def main():
    p = argparse.ArgumentParser(prog="evloop")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--seed", type=int, default=1)
    d.add_argument("--delta", type=float, default=0.5); d.add_argument("--battery-apps", type=int, default=30)
    d.add_argument("--out", default="outputs"); d.set_defaults(f=cmd_demo)
    b = sub.add_parser("benchmark"); b.add_argument("--seeds", type=int, default=12)
    b.add_argument("--deltas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    b.add_argument("--battery-apps", type=int, default=30)
    b.add_argument("--sweep-apps", type=int, nargs="+", default=[15, 50])
    b.add_argument("--workers", type=int, default=2); b.add_argument("--out", default="outputs")
    b.set_defaults(f=cmd_benchmark)
    r = sub.add_parser("report"); r.add_argument("--out", default="outputs")
    r.add_argument("--battery-apps", type=int, default=30); r.set_defaults(f=cmd_report)
    l = sub.add_parser("llm-demo"); l.add_argument("--base-url", required=True); l.add_argument("--model", required=True)
    l.add_argument("--api-key", default=None); l.add_argument("--seed", type=int, default=1)
    l.add_argument("--delta", type=float, default=0.5); l.add_argument("--battery-apps", type=int, default=30)
    l.add_argument("--rounds", type=int, default=1)
    l.add_argument("--no-thinking", action="store_true",
                   help="send chat_template_kwargs.enable_thinking=false (Qwen3-family models on vLLM)")
    l.add_argument("--temperature", type=float, default=0.0, help="negative: do not send a temperature")
    l.add_argument("--top-p", type=float, default=None)
    l.add_argument("--max-tokens-field", default="max_tokens", help="e.g. max_completion_tokens for reasoning APIs")
    l.add_argument("--rho", type=float, default=0.6)
    l.add_argument("--out", default="outputs/llm"); l.set_defaults(f=cmd_llm_demo)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

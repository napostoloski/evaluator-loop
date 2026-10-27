"""Command line.

    python -m evloop demo --seed 1 --delta 0.5
    python -m evloop benchmark --seeds 12 --workers 2   # seeds 100-111
    python -m evloop report
    python -m evloop llm-demo --base-url http://localhost:8000/v1 --model Qwen/Qwen3.8-27B --no-thinking
    python -m evloop app        # local app: simulated panel or your own data (needs pip install -e ".[app]")
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
    extra = json.loads(a.extra_body) if a.extra_body else {}
    if a.no_thinking:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if a.top_p is not None:
        extra["top_p"] = a.top_p
    headers = dict(h.split(":", 1) for h in (a.header or []))
    headers = {k.strip(): v.strip() for k, v in headers.items()}
    client = LLMClient(a.base_url, a.model, api_key=a.api_key, seed=a.seed, extra_body=extra,
                       temperature=None if a.temperature < 0 else a.temperature,
                       max_tokens_field=a.max_tokens_field, extra_headers=headers,
                       cache_dir=os.path.join(a.out, "llm_cache"), log_path=os.path.join(a.out, "llm_calls.jsonl"))
    backend = LLMBackend(client, world, n_twin_pairs=a.twin_pairs, twin=a.twin, verify=not a.no_verify_edits,
                         workers=a.workers, proposer_prompt=a.proposer_prompt, n_examples=a.examples)
    k, budget = _design(a.battery_apps)
    cfg = LoopConfig(seed=a.seed, battery_apps=a.battery_apps, battery_features=k, budget=budget, max_rounds=a.rounds,
                     max_hypotheses=a.hypotheses,
                     test_scope=a.test_scope, het_adjust=a.het_adjust, prior_weight=a.prior_weight)
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
    print(f"\nedits dropped: {backend.dropped_edits} {dict(sorted(backend.drop_reasons.items()))}")
    print(f"model calls: {client.n_calls}; replies cut off without an answer: {client.truncations}")
    for rnd, groups in enumerate(backend.groups, 1):
        pooled = {k: v for k, v in groups.items() if len(v) > 1}
        print(f"round {rnd}: {len(groups)} distinct features after pooling, {len(pooled)} requested by more than one evaluator")
    with open(os.path.join(a.out, f"llm_demo_seed{a.seed}_groups.json"), "w") as f:
        json.dump([{backend.descriptions.get(k, k): v for k, v in g.items()} for g in backend.groups], f, indent=1)


def cmd_audit_edits(a):
    """Replay the rewrite checks over a logged run and report, per library feature, how many
    rewrites the old and new checks accept and how often the accepted ones actually switch
    the benchmark's detector. Needs no model calls."""
    import re

    from .llm import _clean_rewrite, edit_sanity, strip_think, word_similarity
    from .testbed import detect

    def lib(desc):
        """Map a free-text hypothesis to a library detector only when it clearly states that
        feature; returns (feature, inverted) or (None, False)."""
        d = desc.lower()
        if "woman" in d or "female" in d or "male name" in d or "male founder" in d:
            return "gender_cue", ("male name" in d or "male founder" in d) and "female" not in d and "woman" not in d
        if "founder's home" in d or "founder s home" in d or "from home" in d or "home-based" in d:
            return "home_based", False
        if "paying" in d and not re.search(r"figure|survey|interview|research|per unit|cost", d):
            return "traction", "not mention" in d or "does not" in d
        if ("sole founder" in d or "co-founder" in d) and not re.search(r"woman|female|male|man\b", d):
            return "solo_founder", "co-founder" in d
        if "lowercase" in d or "grammatical" in d:
            return "fluency_high", True
        if "united kingdom" in d or "uk university" in d or "uk master" in d:
            return "foreign_degree", False
        if "family business" in d:
            return "family_business", False
        return None, False

    rows = {}
    reasons = {}
    for line in open(a.log):
        c = json.loads(line)
        if "request" not in c:
            continue
        u = c["request"]["messages"][0]["content"]
        if not u.startswith("Rewrite the startup application"):
            continue
        m = re.search(r'becomes (TRUE|FALSE):\s*\n\s*"(.+?)"\s*\n', u, re.S)
        if not m:
            continue
        truth, desc = m.group(1) == "TRUE", m.group(2)
        text = u.split("APPLICATION\n", 1)[1]
        feat, inverted = lib(desc)
        if feat is None:
            continue
        want = int(truth) ^ int(inverted)
        if detect(feat, text) == want:
            continue                                  # no change was needed
        raw = c["content"]
        old_new = strip_think(raw).strip()
        first = text.split(":", 1)[0] + ":"
        pos = old_new.find(first)
        if pos > 0:
            old_new = old_new[pos:]
        old_ok = word_similarity(text, old_new) >= 0.6
        new_text = _clean_rewrite(raw, text)
        ok, reason = edit_sanity(text, new_text)
        if ok and new_text.strip() == text.strip():
            ok, reason = False, "unchanged"
        new_ok = ok and word_similarity(text, new_text) >= 0.6
        if not new_ok:
            reasons[reason or "similarity"] = reasons.get(reason or "similarity", 0) + 1
        r = rows.setdefault(feat, dict(needed=0, old_ok=0, old_flip=0, new_ok=0, new_flip=0))
        r["needed"] += 1
        r["old_ok"] += old_ok
        r["old_flip"] += old_ok and detect(feat, old_new) == want
        r["new_ok"] += new_ok
        r["new_flip"] += new_ok and detect(feat, new_text) == want
    print(f"{'feature':16s} {'needed':>6s} | {'old accepted':>12s} {'of which flip':>13s} | {'new accepted':>12s} {'of which flip':>13s}")
    tot = dict(needed=0, old_ok=0, old_flip=0, new_ok=0, new_flip=0)
    for f, r in sorted(rows.items()):
        for k in tot:
            tot[k] += r[k]
        print(f"{f:16s} {r['needed']:6d} | {r['old_ok']:12d} {r['old_flip']:6d} ({r['old_flip'] / max(r['old_ok'], 1):4.0%}) | "
              f"{r['new_ok']:12d} {r['new_flip']:6d} ({r['new_flip'] / max(r['new_ok'], 1):4.0%})")
    r = tot
    print(f"{'all':16s} {r['needed']:6d} | {r['old_ok']:12d} {r['old_flip']:6d} ({r['old_flip'] / max(r['old_ok'], 1):4.0%}) | "
          f"{r['new_ok']:12d} {r['new_flip']:6d} ({r['new_flip'] / max(r['new_ok'], 1):4.0%})")
    print("rejections by the new checks:", dict(sorted(reasons.items())))


def cmd_app(a):
    import subprocess
    import sys
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit('The app needs Streamlit: pip install -e ".[app]"')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", path, "--server.port", str(a.port)]))


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
    l.add_argument("--header", action="append", help='extra HTTP header, "Name: value" (repeatable)')
    l.add_argument("--twin", choices=["llm", "ridge"], default="llm",
                   help="screen with a few-shot LLM copy (default) or with the ridge text copy on the LLM's edits (no model calls)")
    l.add_argument("--twin-pairs", type=int, default=30, help="edited pairs per hypothesis for the screen")
    l.add_argument("--no-verify-edits", action="store_true", help="skip the verifier call on battery edits")
    l.add_argument("--workers", type=int, default=4, help="concurrent model calls during screening")
    l.add_argument("--test-scope", choices=["proposers", "nominated", "all", "nominated_all"], default="proposers",
                   help="which evaluators are tested on a battery feature (see LoopConfig.test_scope)")
    l.add_argument("--het-adjust", action="store_true", help="allow for real spread of reactions across evaluators")
    l.add_argument("--proposer-prompt", choices=["v1", "v2"], default="v1",
                   help="v2: dimension checklist (founder, venture, evidence, writing), distinct hypotheses")
    l.add_argument("--hypotheses", type=int, default=4, help="hypotheses per evaluator per round")
    l.add_argument("--examples", type=int, default=8, help="over- and under-rated applications shown to the proposer, per side")
    l.add_argument("--prior-weight", type=float, default=1.0,
                   help="with --test-scope all: weighted-BH weight of screened hypotheses relative to unscreened tests")
    l.add_argument("--extra-body", default=None,
                   help='JSON merged into every request, e.g. \'{"thinking": {"type": "between_tools"}}\' for Claude')
    l.add_argument("--out", default="outputs/llm"); l.set_defaults(f=cmd_llm_demo)
    au = sub.add_parser("audit-edits", help="replay the rewrite checks over a logged LLM run (no model calls)")
    au.add_argument("--log", required=True, help="llm_calls.jsonl from a run")
    au.set_defaults(f=cmd_audit_edits)
    ap = sub.add_parser("app", help="local app: run the prototype on a simulated panel or on your own data")
    ap.add_argument("--port", type=int, default=8501); ap.set_defaults(f=cmd_app)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()

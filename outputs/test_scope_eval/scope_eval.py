"""Compare test scopes on the offline benchmark. Usage: scope_eval.py OUT.csv seed [seed ...]"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from evloop.evaluators import make_world
from evloop.loop import LoopConfig, TextSpace, run_panel
from evloop.testbed import sample_applications

VARIANTS = {
    "proposers": dict(test_scope="proposers"),
    "nominated": dict(test_scope="nominated"),
    "all": dict(test_scope="all"),
    "all_het": dict(test_scope="all", het_adjust=True),
    "nominated_het": dict(test_scope="nominated", het_adjust=True),
}
SHARED = {"traction", "fluency_high", "experienced", "solo_founder"}

def job(args):
    seed, delta = args
    t0 = time.time()
    apps = sample_applications(156, seed=seed)
    space = TextSpace(apps, seed=seed)
    rows = []
    for name, kw in VARIANTS.items():
        world = make_world(apps, seed=seed, delta=delta)
        cfg = LoopConfig(seed=seed, battery_apps=30, battery_features=5, budget=150, **kw)
        res = run_panel(world, space, cfg)
        for j, ev in enumerate(world.panel):
            planted = {r.key for r in ev.rules}
            got = {(f, t) for f, t, _ in res[j].accepted}
            fps = sorted(got - planted)
            rows.append(dict(seed=seed, delta=delta, variant=name, eid=j, role=ev.role,
                             tp=len(got & planted), fp=len(fps), fn=len(planted - got),
                             fp_shared=sum(1 for f, _ in fps if f in SHARED),
                             n_tests=sum(1 for h in res[j].trail if h.tested),
                             queries=res[j].queries, fps=json.dumps(fps)))
    return rows, round(time.time() - t0, 1)

if __name__ == "__main__":
    out = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]]
    deltas = [0.5]
    jobs = [(s, d) for s in seeds for d in deltas]
    allrows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for rows, secs in ex.map(job, jobs):
            allrows += rows
            print(f"seed {rows[0]['seed']} done in {secs}s", flush=True)
            pd.DataFrame(allrows).to_csv(out, index=False)

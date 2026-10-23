"""Offline: text-lasso proposer vs coded-feature proposer, full loop. Usage: proposer_eval.py OUT.csv seed ..."""
import sys, json, time
sys.path.insert(0, ".")
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from evloop.evaluators import make_world
from evloop.loop import LoopConfig, TextSpace, OfflineBackend, run_panel
from evloop.testbed import sample_applications

VARIANTS = {"text": dict(proposer="text"), "coded_p20": dict(proposer="coded", p_max=0.2),
            "coded_p10": dict(proposer="coded", p_max=0.1), "coded_p35": dict(proposer="coded", p_max=0.35)}

def job(seed):
    t0 = time.time()
    apps = sample_applications(156, seed=seed)
    space = TextSpace(apps, seed=seed)
    rows = []
    for name, kw in VARIANTS.items():
        world = make_world(apps, seed=seed, delta=0.5)
        cfg = LoopConfig(seed=seed, battery_apps=30, battery_features=5, budget=150)
        res = run_panel(world, OfflineBackend(space, **kw), cfg)
        for j, ev in enumerate(world.panel):
            planted = {r.key for r in ev.rules}
            got = {(f, t) for f, t, _ in res[j].accepted}
            prop1 = {(h.feature, h.target) for h in res[j].trail if h.round == 1 and h.source == "proposed"}
            passed1 = {(h.feature, h.target) for h in res[j].trail if h.round == 1 and h.twin_pass}
            rows.append(dict(seed=seed, variant=name, eid=j, role=ev.role, tp=len(got & planted),
                             fp=len(got - planted), fn=len(planted - got), n_planted=len(planted),
                             proposed_hit=len(prop1 & planted), passed_hit=len(passed1 & planted),
                             n_proposed=len(prop1), fps=json.dumps(sorted(got - planted))))
    return rows, round(time.time() - t0, 1)

if __name__ == "__main__":
    out = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]]
    allrows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for rows, secs in ex.map(job, seeds):
            allrows += rows
            print(f"seed {rows[0]['seed']} done in {secs}s", flush=True)
            pd.DataFrame(allrows).to_csv(out, index=False)

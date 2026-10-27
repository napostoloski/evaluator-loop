"""Pre-compute offline prototype runs over a grid of settings for the web page.

    python scripts/precompute_runs.py --out outputs/interface --workers 2

Writes one JSON summary per run (outputs/interface/runs/). Each run takes about 20 seconds.
Then: python scripts/build_explorer_data.py outputs/interface explorer/runs.json
"""
import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor


def one(args):
    delta, rho, battery, seed, out = args
    path = os.path.join(out, "runs", f"d{delta}_r{rho}_b{battery}_s{seed}.json")
    if os.path.exists(path):
        return path
    from evloop.evaluators import make_world
    from evloop.loop import LoopConfig, TextSpace, run_panel
    from evloop.present import simulated_summary
    from evloop.testbed import sample_applications
    k = max(1, min(5, 150 // battery))
    apps = sample_applications(156, seed=seed)
    world = make_world(apps, seed=seed, delta=delta, rho=rho)
    cfg = LoopConfig(seed=seed, battery_apps=battery, battery_features=k, budget=k * battery)
    res = run_panel(world, TextSpace(apps, seed=seed), cfg)
    s = simulated_summary(world, res, mode="offline",
                          settings=dict(delta=delta, rho=rho, battery=battery, seed=seed))
    with open(path, "w") as f:
        json.dump(s, f)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/interface")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    grid = list(itertools.product([0.25, 0.5, 0.75], [0.3, 0.6, 0.9], [15, 30, 50], [1, 2, 3]))
    with ProcessPoolExecutor(a.workers) as ex:
        for i, p in enumerate(ex.map(one, [(d, r, b, s, a.out) for d, r, b, s in grid]), 1):
            print(f"{i}/{len(grid)} {os.path.basename(p)}", flush=True)
    print("done; build the explorer data with scripts/build_explorer_data.py")


if __name__ == "__main__":
    main()

"""Combine the pre-computed runs and the real-AI run into the web explorer's data file.

    python scripts/build_explorer_data.py outputs/interface ../explorer/runs.json
"""
import glob
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
grid = {}
for p in glob.glob(os.path.join(src, "runs", "*.json")):
    with open(p) as f:
        r = json.load(f)
    s = r["settings"]
    grid[f"{s['delta']}|{s['rho']}|{s['battery']}|{s['seed']}"] = r
with open(os.path.join(src, "extra.json")) as f:
    extra = json.load(f)
with open(dst, "w") as f:
    json.dump(dict(grid=grid, ai=extra["ai"], example_application=extra["example_application"]), f, separators=(",", ":"))
print(f"{len(grid)} runs -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")

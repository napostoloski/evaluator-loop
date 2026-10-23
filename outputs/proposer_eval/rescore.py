import sys, json, importlib.util, pandas as pd
spec = importlib.util.spec_from_file_location("pe", sys.argv[2]); src = open(sys.argv[2]).read()
ns = {}; exec(src[src.index("import sys"):src.index("OUT = sys.argv")], ns); exec(src[src.index("TECH ="):src.index("rows, texts_out")], ns)
d = pd.read_csv(sys.argv[1], keep_default_na=False)
hits = []
for _, r in d.iterrows():
    planted = {tuple(x) for x in json.loads(r.planted)}
    keys = {ns["classify"](desc, t) for desc, t, _ in json.loads(r.proposed)}
    hits.append(len(planted & keys))
d["hits2"] = hits
print(d.groupby(["seed","variant"]).agg(hits=("hits2","sum"), planted=("n_planted","sum")).unstack("variant"))
tot = d.groupby("variant").agg(hits=("hits2","sum"), planted=("n_planted","sum")); tot["rate"] = tot.hits/tot.planted; print(tot)

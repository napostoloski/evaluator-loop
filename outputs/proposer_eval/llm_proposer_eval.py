"""Proposer-only test on the real model: current prompt (v1: 8 examples, k=4) vs revised (v2: 20 examples, k=6).
Scores whether each planted preference appears among the evaluator's hypotheses. Hard cost cap."""
import sys, os, re, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from evloop.evaluators import make_world
from evloop.llm import LLMClient, propose_llm
from evloop.loop import consensus_residual
from evloop.testbed import sample_applications

OUT = sys.argv[1]; SEEDS = [int(x) for x in sys.argv[2].split(",")]; CAP = float(sys.argv[3])
VARIANTS = {"v1": dict(k=4, n_examples=8, prompt="v1"), "v2": dict(k=6, n_examples=20, prompt="v2")}
def make_client(seed):
    return LLMClient("https://api.anthropic.com/v1", "claude-sonnet-5-5", temperature=None, seed=seed,
                   extra_body={"thinking": {"type": "between_tools"}},
                     cache_dir="outputs/proposer_test/llm_cache", log_path="outputs/proposer_test/llm_calls.jsonl")
client = make_client(1)

def spent():
    if not os.path.exists(client.log_path): return 0.0
    t = 0.0
    for l in open(client.log_path):
        c = json.loads(l)
        if not c.get("cached") and c.get("usage"):
            t += c["usage"]["prompt_tokens"] * 2e-6 + c["usage"]["completion_tokens"] * 10e-6
    return t

TECH = r"software|platform|technology|tech\b|app\b|apps\b|digital|fintech|edtech|health-tech|cloud|online tutoring|telemedicine"
CONS = r"clothing|textile|food|meal|furniture|handicraft|craft|apparel|kurta|consumer|physical|skincare|decor|garment|tailor"
def classify(desc, target):
    d = desc.lower()
    subj = re.split(r"rather than|instead of|as opposed to|whereas", d)[0]     # the property itself, not the contrast
    # writing first: grammar hypotheses quote 'he have', which must not read as gender
    if re.search(r"lowercase|grammar|grammatical|punctuation|typo|spelling|sloppy|slips|capitali|agreement|unpolished|fluent|polished|well-written", d):
        good = re.search(r"standard|correct|fluent|polished|well-written|proper|no (?:grammar|spelling)|clean", subj) and not re.search(r"error|mistake|lack|missing|lowercase|slip|sloppy|unpolished", subj)
        return ("fluency_high", target if good else 1 - target)
    if re.search(r"\b(woman|women|female|she|her)\b", subj) and not re.search(r"\b(man|men|male|he|his)\b", subj): return ("gender_cue", target)
    if re.search(r"\b(man|men|male|he|his)\b", subj) and not re.search(r"\b(woman|women|female|she|her)\b", subj): return ("gender_cue", 1 - target)
    if re.search(r"founder'?s? home|from home|out of (?:her|his|their|the founder'?s?) home|home-based|at home", subj):
        return ("home_based", 1 - target if re.search(r"not run|does not|no longer", subj) else target)
    if re.search(r"paying|pilot customers|revenue", subj) and not re.search(r"survey|interview|per-unit|per unit|cost", subj):
        return ("traction", 1 - target if re.search(r"does not|no paying|without|not mention", subj) else target)
    t, c = re.search(TECH, subj), re.search(CONS, subj)
    if t or c:
        first_tech = t and (not c or t.start() < c.start())
        return ("consumer_sector", 1 - target if first_tech else target)
    return ("other", target)

rows, texts_out = [], []
for seed in SEEDS:
    client = make_client(seed)   # same request (incl. seed) as llm-demo, so seed-1 v1 calls hit the cache
    apps = sample_applications(156, seed=seed); w = make_world(apps, seed=seed, delta=0.5)
    texts = [a.text for a in apps]
    for j, ev in enumerate(w.panel):
        resid, _ = consensus_residual(w.scores, j)
        planted = {r.key for r in ev.rules}
        for v, kw in VARIANTS.items():
            if spent() > CAP:
                print(f"COST CAP ${CAP} reached at seed {seed} evaluator {j}; stopping", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False); sys.exit(0)
            hyps = propose_llm(client, texts, resid, **kw)
            keys = [classify(h["description"], h["target"]) for h in hyps]
            hit = planted & set(keys)
            rows.append(dict(seed=seed, eid=j, role=ev.role, variant=v, n_planted=len(planted), hits=len(hit),
                             n_hyps=len(hyps), n_distinct=len(set(keys)), planted=json.dumps(sorted(planted)),
                             proposed=json.dumps([[h["description"], h["target"], list(k)] for h, k in zip(hyps, keys)])))
        print(f"seed {seed} evaluator {j} done; spent ${spent():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"DONE; spent ${spent():.2f}")

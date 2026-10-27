"""Plain-language summaries of prototype runs, shared by the local app and the web page.

simulated_summary() turns a run on a synthetic panel into a JSON-ready dict: for every
evaluator, the planted preferences (the truth), every hypothesis the prototype considered
with its screening and re-rating results, and whether each confirmed preference was real.
"""
from __future__ import annotations

import math
import re

import numpy as np

# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------
RULE_TEXT = {
    ("gender_cue", 1): "favours women founders (name and pronouns)",
    ("gender_cue", 0): "favours men founders (name and pronouns)",
    ("consumer_sector", 1): "favours consumer and lifestyle ventures",
    ("consumer_sector", 0): "favours technology ventures",
    ("home_based", 1): "favours businesses run from home",
    ("home_based", 0): "penalises businesses run from home",
    ("fluency_high", 1): "favours fluent, error-free English",
    ("fluency_high", 0): "favours applications with writing slips",
    ("traction", 1): "favours reported paying pilot customers",
    ("traction", 0): "penalises reported paying pilot customers",
    ("experienced", 1): "favours founders with five or more years of experience",
    ("experienced", 0): "favours less experienced founders",
    ("solo_founder", 1): "favours solo founders",
    ("solo_founder", 0): "favours teams over solo founders",
    ("metro_city", 1): "favours ventures in large cities",
    ("metro_city", 0): "favours ventures outside large cities",
    ("family_business", 1): "favours founders from a family business",
    ("family_business", 0): "penalises founders from a family business",
    ("foreign_degree", 1): "favours founders with a foreign degree",
    ("foreign_degree", 0): "penalises founders with a foreign degree",
}

ROLE_TEXT = {
    "cue": "Same-gender premium, triggered by the founder's name and pronouns",
    "proxy": "Prefers a sector that is more common among founders of their own gender",
    "proxy_home": "Reacts to home-based businesses, which are more common among women founders",
    "fluency": "Prefers fluent English",
    "ability": "Prefers reported paying customers",
    "multi": "Same-gender premium and a preference for paying customers",
    "null": "No planted preference",
}

SETTINGS = [
    dict(key="delta", label="Size of the planted preferences", options=[0.25, 0.5, 0.75], default=0.5,
         unit="points on the 1-5 scale",
         help="How much a planted preference moves an evaluator's score. 0.25 is subtle, 0.5 moderate, "
              "0.75 strong. Scores average five criteria rated 1 to 5, so a half-point preference shifts "
              "the evaluator's average score by half a point."),
    dict(key="rho", label="Consistency of re-ratings", options=[0.3, 0.6, 0.9], default=0.6, unit="correlation",
         help="How similarly an evaluator reads the same venture when rating it a second time. At 0.9 the "
              "second reading is almost the same as the first; at 0.3 much of it is new. Less consistent "
              "re-ratings are noisier and make preferences harder to confirm."),
    dict(key="battery", label="Re-ratings per hypothesis", options=[15, 30, 50], default=30, unit="applications",
         help="How many edited applications each evaluator re-rates to test one hypothesis, half with the "
              "feature added and half with it removed. The prototype tests up to five hypotheses per round "
              "within a cap of 150 re-ratings per evaluator: 15 per hypothesis uses 75 re-ratings, 30 uses "
              "150, and 50 allows only three tests."),
    dict(key="seed", label="Simulated panel", options=[1, 2, 3], default=1, unit="",
         help="Which random panel to draw. Every panel has the same structure but newly drawn applications "
              "and evaluators, so comparing panels shows how much results vary by chance."),
]

HOW_APPLICATIONS = (
    "Each simulated panel has 156 applications. Every application is generated from a set of founder and "
    "venture attributes and written out as a short text with sections on the team, the venture and the "
    "evidence, modelled on the XSEED application form. The attributes are the founder's gender (40% women), "
    "shown through the name and pronouns; the city (60% in a large metropolitan city); the sector (consumer "
    "and lifestyle, or technology); whether the business runs from home; how fluent the English is (low "
    "fluency produces small writing slips); whether the founder has a foreign degree or grew up in a family "
    "business; team size (35% apply alone); years of experience; and a hidden quality score that drives the "
    "venture's stage and whether it reports paying pilot customers. Several attributes travel together on "
    "purpose. Women founders more often run consumer ventures (65% against 30%) and home-based businesses, "
    "and founders in large cities more often write fluent English and hold foreign degrees. These "
    "correlations are what make the true cause of a judgment hard to find."
)

HOW_EVALUATORS = (
    "Each panel has 20 evaluators, 10 women and 10 men, and every evaluator scores every application on five "
    "criteria from 1 to 5. An evaluator's score combines their leniency, how well they read the hidden "
    "quality, their own noisy reading of each venture, reactions the whole panel shares (paying customers "
    "+0.30, fluent English +0.15, experience +0.15, applying alone -0.15, each varying slightly from "
    "evaluator to evaluator), and any planted preferences. Among the 10 evaluators of each gender, two have a "
    "same-gender premium triggered by the founder's name and pronouns, one prefers a sector that is more "
    "common among founders of their own gender, one reacts to home-based businesses (women evaluators favour "
    "them, men penalise them), one prefers fluent English, one prefers paying customers, one has both the "
    "same-gender premium and the preference for paying customers, and three have no planted preference. "
    "That makes 16 planted preferences across 14 evaluators. The prototype is never told any of this."
)

HOW_PROTOTYPE = [
    ("Spot the disagreements", "For each evaluator, measure how far their score for each application departs "
                               "from what the rest of the panel gave it."),
    ("Propose explanations", "Find features of the text that could explain the departures. Offline this is a "
                             "lasso over the text; with an AI model, the model reads the most over- and "
                             "under-rated applications and states features in plain words."),
    ("Build the tests", "Rewrite applications so that only that one feature changes. Offline the rewrites come "
                        "from a fixed library of edits; with an AI model, the model writes them and automatic "
                        "checks discard edits that change anything else."),
    ("Screen cheaply", "A statistical copy of the evaluator, trained on their past scores, predicts whether the "
                       "edits would move them. Only ideas the copy supports go forward."),
    ("Confirm with re-ratings", "The evaluator re-rates edited applications. An idea counts only if this "
                                "evaluator reacts more than the rest of the panel, after correcting for the number "
                                "of tests. In the simulation the re-ratings come from the simulated evaluators."),
    ("Repeat", "Confirmed preferences are removed and the search runs again on what is left."),
]

UPLOAD_FORMAT = [
    ("applications.csv", "application_id, text, founder_gender (optional: F or M)"),
    ("ratings.csv", "evaluator_id, application_id, score (one row per rating)"),
    ("evaluators.csv", "evaluator_id, gender (optional: F or M); the whole file is optional"),
]


def rule_text(feature, target):
    return RULE_TEXT.get((feature, int(target)), f"{feature} = {target}")


def _num(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def match_library(desc: str):
    """Map an AI model's free-text hypothesis to a simulation feature when it clearly states
    one. Returns (feature, inverted) or (None, False). Used only to score simulated runs."""
    d = (desc or "").lower()
    if "woman" in d or "female" in d or "male name" in d or "male founder" in d or re.search(r"\bmen\b|\bman\b", d):
        inv = ("male name" in d or "male founder" in d or re.search(r"\bman\b|\bmen\b", d) is not None) \
            and "female" not in d and "woman" not in d
        return "gender_cue", bool(inv)
    if "founder's home" in d or "founder s home" in d or "from home" in d or "home-based" in d:
        return "home_based", False
    if "paying" in d and not re.search(r"figure|survey|interview|research|per unit|cost", d):
        return "traction", "not mention" in d or "does not" in d
    if ("sole founder" in d or "co-founder" in d or "solo" in d) and not re.search(r"woman|female|male|man\b", d):
        return "solo_founder", "co-founder" in d
    if "lowercase" in d or "grammatical" in d or "fluent" in d or "error-free" in d:
        return "fluency_high", "error" in d and "error-free" not in d
    if "consumer" in d or "lifestyle" in d:
        return "consumer_sector", False
    if "technology" in d or "software" in d:
        return "consumer_sector", True
    if "united kingdom" in d or "uk university" in d or "foreign" in d or "abroad" in d:
        return "foreign_degree", False
    if "family business" in d:
        return "family_business", False
    if "years of experience" in d or "experienced" in d:
        return "experienced", False
    return None, False


def _verdict(h):
    if not h.get("screen_pass"):
        return "dropped at screening"
    if not h.get("tested"):
        return "passed screening, not tested"
    return "confirmed" if h.get("accepted") else "tested, not confirmed"


def simulated_summary(world, results, mode: str = "offline", settings: dict | None = None,
                      label: str = "") -> dict:
    """JSON-ready summary of a run on a synthetic panel.

    results: {eid: EvaluatorResult} from loop.run_panel. In mode "offline" hypotheses are library
    features; in mode "ai" they are free text, and match_library() decides whether a confirmed
    hypothesis is a planted one.
    """
    evaluators = []
    planted_total = found = reported = real_reported = flagged_null = n_null = 0
    for j, ev in enumerate(world.panel):
        res = results[j]
        planted = {(r.feature, int(r.target)) for r in ev.rules}
        hyps, hit = [], set()
        for h in res.trail:
            d = dict(text=h.description or rule_text(h.feature, h.target),
                     direction="rewards" if h.target == 1 else "penalises",
                     round=int(h.round), source=h.source,
                     screen_effect=_num(h.twin_effect), screen_low=_num(h.twin_ci[0]), screen_high=_num(h.twin_ci[1]),
                     screen_pass=bool(h.twin_pass), tested=bool(h.tested),
                     effect=_num(h.human_effect), se=_num(h.human_se), q=_num(h.human_q, 4),
                     accepted=bool(h.accepted))
            if mode == "offline":
                key = (h.feature, int(h.target))
                d["text"] = rule_text(*key)
            else:
                f, inv = match_library(h.description)
                key = (f, int(h.target) ^ int(inv)) if f else None
            d["planted"] = bool(key in planted) if key else False
            if h.accepted:
                reported += 1
                if d["planted"]:
                    real_reported += 1
                    hit.add(key)
            d["verdict"] = _verdict(d)
            hyps.append(d)
        planted_total += len(planted)
        found += len(hit)
        if not planted:
            n_null += 1
            flagged_null += int(any(h["accepted"] for h in hyps))
        evaluators.append(dict(
            id=j, gender=ev.gender, role=ev.role, role_text=ROLE_TEXT.get(ev.role, ev.role),
            planted=[dict(text=rule_text(f, t), found=(f, t) in hit) for f, t in sorted(planted)],
            hypotheses=hyps,
            confirmed=[h["text"] for h in hyps if h["accepted"]],
            rerated=int(res.queries),
        ))
    summary = dict(planted=planted_total, found=found, reported=reported, real=real_reported,
                   real_share=_num(real_reported / reported if reported else float("nan"), 3),
                   null_evaluators=n_null, null_flagged=flagged_null)
    return dict(label=label, mode=mode, settings=settings or {}, summary=summary, evaluators=evaluators)


def summary_from_trail(world, trail_df, settings: dict | None = None, label: str = "") -> dict:
    """Rebuild an AI-mode summary from a saved audit trail (no model calls)."""
    from .loop import EvaluatorResult, Hypothesis
    results = {j: EvaluatorResult(eid=j) for j in range(len(world.panel))}
    for r in trail_df.to_dict("records"):
        h = Hypothesis(str(r["feature"]), int(r["target"]), float("nan"), [], int(r["round"]),
                       description=str(r["description"]))
        h.twin_effect = float(r["twin_effect"]) if r["twin_effect"] == r["twin_effect"] else float("nan")
        h.twin_ci = (float(r["twin_ci_lo"]), float(r["twin_ci_hi"]))
        h.twin_pass = str(r["twin_pass"]) == "True"
        h.tested = str(r["tested"]) == "True"
        for k in ("human_effect", "human_se", "human_q"):
            setattr(h, k, float(r[k]) if r[k] == r[k] and r[k] != "" else float("nan"))
        h.accepted = str(r["accepted"]) == "True"
        h.source = str(r.get("source", "proposed"))
        results[int(r["eid"])].trail.append(h)
    # every evaluator re-rates every battery, so the re-ratings are the same for all of them
    sizes = {}
    for r in trail_df.to_dict("records"):
        if str(r["tested"]) == "True" and r["battery_apps"] == r["battery_apps"]:
            sizes[str(r["feature"])] = max(sizes.get(str(r["feature"]), 0), int(float(r["battery_apps"])))
    for res in results.values():
        res.queries = sum(sizes.values())
    return simulated_summary(world, results, mode="ai", settings=settings, label=label)


def upload_summary(data, sr) -> dict:
    """JSON-ready summary of a screening run on uploaded data."""
    evaluators = []
    for j, res in sr.results.items():
        prem, p = sr.premium.get(j, (float("nan"), float("nan")))
        hyps = []
        for h in res.trail:
            hyps.append(dict(text=h.description, direction="rewards" if h.target == 1 else "penalises",
                             screen_effect=_num(h.twin_effect), screen_low=_num(h.twin_ci[0]),
                             screen_high=_num(h.twin_ci[1]), screen_pass=bool(h.twin_pass), pairs=int(h.n_pairs),
                             examples=[dict(with_feature=a, without_feature=b) for a, b in h.examples[:2]],
                             verdict="passed screening" if h.twin_pass else "dropped at screening"))
        evaluators.append(dict(id=data.panel[j].label, gender=data.panel[j].gender,
                               premium=_num(prem), premium_p=_num(p, 4), hypotheses=hyps))
    feats = [dict(text=v["description"], evaluators=[data.panel[j].label for j in v["evaluators"]])
             for v in sr.features.values()]
    return dict(mode="upload", notes=sr.notes, evaluators=evaluators, shared_features=feats,
                edits_to_test=len(sr.batteries))

"""Uploaded data: a researcher's own applications and ratings.

With real data there are no re-ratings yet, so the loop runs up to screening: for each
evaluator it measures how far their scores depart from the rest of the panel, a language
model proposes features of the text that could explain the departure, the model rewrites
applications so that only that feature changes, and a statistical copy of the evaluator
predicts whether the edits would move them. The output is a set of screened hypotheses per
evaluator and, for the most requested features, the edited applications that the evaluators
would re-rate in a confirmation study. Nothing here is a confirmed finding.

Input format (CSV files or pandas DataFrames):

    applications: application_id, text [, founder_gender]
    ratings:      evaluator_id, application_id, score        (one row per rating)
    evaluators:   evaluator_id [, gender]                    (optional)

Gender columns are optional and only used to report each evaluator's same-gender premium;
use F and M. Every evaluator should rate every application. A few missing ratings are
filled so that they carry no information (see load_dataset).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .loop import EvaluatorResult, LoopConfig, TextSpace, consensus_residual, premium

MIN_APPS = 20
MIN_EVALUATORS = 4
MAX_MISSING_SHARE = 0.2


@dataclass
class TextApp:
    app_id: int
    text: str
    label: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Rater:
    eid: int
    label: str
    gender: str | None = None


@dataclass
class Dataset:
    apps: list
    panel: list
    scores: np.ndarray            # evaluators x applications
    notes: list = field(default_factory=list)


class DataError(ValueError):
    """Raised with a message meant for the researcher when an upload cannot be used."""


def _frame(x, name):
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x.copy()
    try:
        return pd.read_csv(x, dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        raise DataError(f"Could not read the {name} file as CSV: {e}") from e


def _require(df, cols, name):
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"The {name} file needs the column(s) {', '.join(missing)}; "
                        f"it has {', '.join(df.columns) or 'none'}.")
    return df


def _gender(v):
    v = str(v).strip().upper()
    return {"F": "F", "FEMALE": "F", "WOMAN": "F", "W": "F", "M": "M", "MALE": "M", "MAN": "M"}.get(v)


def load_dataset(applications, ratings, evaluators=None) -> Dataset:
    """Build a Dataset from the three inputs, checking them and explaining any problem.

    Missing ratings: an evaluator who rated fewer than 80% of the applications is dropped.
    Remaining gaps are filled with the other evaluators' mean for that application plus the
    evaluator's own average departure from them, so a filled cell has no departure of its
    own and cannot drive a hypothesis.
    """
    A = _require(_frame(applications, "applications"), ["application_id", "text"], "applications")
    R = _require(_frame(ratings, "ratings"), ["evaluator_id", "application_id", "score"], "ratings")
    V = _frame(evaluators, "evaluators")
    if V is not None:
        V = _require(V, ["evaluator_id"], "evaluators")

    notes = []
    A["application_id"] = A["application_id"].astype(str).str.strip()
    A["text"] = A["text"].astype(str)
    if A["application_id"].duplicated().any():
        raise DataError("Each application_id must appear once in the applications file.")
    empty = A["text"].str.strip() == ""
    if empty.any():
        notes.append(f"{int(empty.sum())} application(s) with no text were left out.")
        A = A[~empty]
    if len(A) < MIN_APPS:
        raise DataError(f"At least {MIN_APPS} applications are needed; the file has {len(A)}.")

    R["evaluator_id"] = R["evaluator_id"].astype(str).str.strip()
    R["application_id"] = R["application_id"].astype(str).str.strip()
    R["score"] = pd.to_numeric(R["score"], errors="coerce")
    bad = R["score"].isna().sum()
    if bad:
        notes.append(f"{int(bad)} rating(s) that are not numbers were ignored.")
        R = R.dropna(subset=["score"])
    unknown = ~R["application_id"].isin(A["application_id"])
    if unknown.any():
        notes.append(f"{int(unknown.sum())} rating(s) refer to applications not in the applications file and were ignored.")
        R = R[~unknown]
    if R.duplicated(["evaluator_id", "application_id"]).any():
        notes.append("Some evaluators rated the same application more than once; their ratings were averaged.")
    W = R.pivot_table(index="evaluator_id", columns="application_id", values="score", aggfunc="mean")
    W = W.reindex(columns=list(A["application_id"]))

    coverage = W.notna().mean(axis=1)
    low = coverage[coverage < 1 - MAX_MISSING_SHARE]
    if len(low):
        notes.append(f"{len(low)} evaluator(s) rated fewer than {int(100 * (1 - MAX_MISSING_SHARE))}% of the "
                     f"applications and were left out: {', '.join(low.index)}.")
        W = W.drop(index=low.index)
    if len(W) < MIN_EVALUATORS:
        raise DataError(f"At least {MIN_EVALUATORS} evaluators who rated most applications are needed; "
                        f"there are {len(W)}.")
    rated = W.notna().any(axis=0)
    if not rated.all():
        notes.append(f"{int((~rated).sum())} application(s) that no remaining evaluator rated were left out.")
        W = W.loc[:, rated]
        A = A[A["application_id"].isin(W.columns)]

    S = np.array(W.to_numpy(dtype=float), copy=True)
    n_missing = int(np.isnan(S).sum())
    if n_missing:
        for j in range(S.shape[0]):
            miss = np.isnan(S[j])
            if not miss.any():
                continue
            others = np.nanmean(np.delete(S, j, axis=0), axis=0)
            offset = np.nanmean(S[j] - others)
            S[j, miss] = others[miss] + offset
        notes.append(f"{n_missing} missing rating(s) were filled so that they carry no information.")

    gender = {}
    if V is not None and "gender" in V.columns:
        gender = {str(r.evaluator_id).strip(): _gender(r.gender) for r in V.itertuples()}
    apps = []
    for i, r in enumerate(A.itertuples()):
        attrs = {}
        if "founder_gender" in A.columns and _gender(r.founder_gender):
            attrs["gender"] = _gender(r.founder_gender)
        apps.append(TextApp(i, r.text, r.application_id, attrs))
    panel = [Rater(j, str(e), gender.get(str(e))) for j, e in enumerate(W.index)]
    return Dataset(apps, panel, S, notes)


def has_gender(data: Dataset) -> bool:
    return all("gender" in a.attrs for a in data.apps) and all(r.gender for r in data.panel)


def same_gender_premium(data: Dataset, j: int, resid) -> tuple:
    """(premium, p) of evaluator j for founders of their own gender, or (nan, nan)."""
    if not has_gender(data):
        return float("nan"), float("nan")
    fem = np.array([a.attrs["gender"] == "F" for a in data.apps])
    same = fem if data.panel[j].gender == "F" else ~fem
    if same.all() or not same.any():
        return float("nan"), float("nan")
    return premium(resid, same)


def quick_candidates(space: TextSpace, resid, k: int = 6, min_share: float = 0.05) -> dict:
    """Words and phrases most associated with an evaluator's departures from the panel,
    without any language model. These are correlations, not tested hypotheses.
    Returns {"higher": [(term, r, p)], "lower": [...]}.
    """
    n_words = len(space.vec.get_feature_names_out())
    T = space.vec.transform(space.texts)
    present = (T > 0).toarray().astype(float)
    share = present.mean(0)
    keep = np.where((share >= min_share) & (share <= 1 - min_share))[0]
    out = {"higher": [], "lower": []}
    if len(keep) == 0:
        return out
    rows = []
    for c in keep[:n_words]:
        x = present[:, c]
        r, p = stats.pearsonr(x, resid)
        rows.append((space.vocab[c], float(r), float(p)))
    rows.sort(key=lambda t: t[1])
    out["lower"] = rows[:k]
    out["higher"] = rows[::-1][:k]
    return out


@dataclass
class ScreenResult:
    results: dict                 # eid -> EvaluatorResult (trail holds the hypotheses)
    resid: dict                   # eid -> departure from the panel on each application
    premium: dict                 # eid -> (same-gender premium, p) or (nan, nan)
    batteries: list               # edited applications to re-rate, one dict per edit
    features: dict                # feature key -> {"description", "evaluators"}
    notes: list


def screen_dataset(data: Dataset, backend, cfg: LoopConfig | None = None, design_batteries: bool = True,
                   progress=None, log=print) -> ScreenResult:
    """Observe, propose, edit and screen for every evaluator; then, for the features that
    passed screening for the most evaluators, prepare the edited applications for a
    re-rating battery (counterbalanced: half with the feature switched on, half off).
    `backend` is an evloop.llm.LLMBackend built on `data`."""
    cfg = cfg or LoopConfig()
    E = len(data.panel)
    results, resids, prem = {}, {}, {}
    for j in range(E):
        rng = np.random.default_rng(cfg.seed * 1000 + 100 + j)
        resid, _ = consensus_residual(data.scores, j)
        resids[j] = resid
        prem[j] = same_gender_premium(data, j, resid)
        hyps = backend.propose(j, resid, set(), cfg, rng, 1)
        if hyps:
            backend.screen(j, resid, hyps, cfg, rng)
            for h in hyps:
                h.twin_pass = bool(h.twin_ci[0] > 0 and h.twin_effect >= cfg.min_twin_effect)
        results[j] = EvaluatorResult(eid=j, trail=list(hyps))
        if progress:
            progress((j + 1) / (E + (1 if design_batteries else 0)), f"Evaluator {data.panel[j].label}: "
                     f"{sum(h.twin_pass for h in hyps)} of {len(hyps)} hypotheses passed screening")

    cands = {j: [h for h in r.trail if h.twin_pass] for j, r in results.items()}
    cands = {j: hs for j, hs in cands.items() if hs}
    if len(cands) and hasattr(backend, "canonicalize"):
        backend.canonicalize(cands)
    features = {}
    for j, hs in cands.items():
        for h in hs:
            f = features.setdefault(h.feature, {"description": h.description, "evaluators": [], "best": 0.0})
            if j not in f["evaluators"]:
                f["evaluators"].append(j)
            f["best"] = max(f["best"], h.twin_effect)

    batteries = []
    if design_batteries and features:
        order = sorted(features, key=lambda k: (-len(features[k]["evaluators"]), -features[k]["best"]))
        rng = np.random.default_rng(cfg.seed * 1000 + 7)
        used = set()
        half = cfg.battery_apps // 2
        for f in order[: cfg.battery_features]:
            flip = backend.flip_fn(f, features[f]["description"])
            pos, neg = [], []
            for i in rng.permutation(len(data.apps)):
                i = int(i)
                if i in used:
                    continue
                if len(pos) >= half and len(neg) >= cfg.battery_apps - half:
                    break
                out = flip(data.apps[i])
                if out is None:
                    continue
                if out[1] > 0 and len(pos) < half:
                    pos.append((i, out[0], "added"))
                elif out[1] < 0 and len(neg) < cfg.battery_apps - half:
                    neg.append((i, out[0], "removed"))
            for i, text, how in pos + neg:
                used.add(i)
                batteries.append(dict(feature=f, description=features[f]["description"],
                                      requested_by=[data.panel[j].label for j in features[f]["evaluators"]],
                                      application_id=data.apps[i].label, change=how,
                                      original_text=data.apps[i].text, edited_text=text))
            log(f"battery for '{features[f]['description'][:60]}': {len(pos)} added, {len(neg)} removed")
        if progress:
            progress(1.0, "Edited applications prepared for re-rating")
    return ScreenResult(results, resids, prem, batteries, features, list(data.notes))


def hypotheses_table(data: Dataset, sr: ScreenResult) -> pd.DataFrame:
    rows = []
    for j, res in sr.results.items():
        for h in res.trail:
            rows.append(dict(evaluator_id=data.panel[j].label,
                             hypothesis=h.description,
                             direction="rewards" if h.target == 1 else "penalises",
                             predicted_effect=round(h.twin_effect, 3) if np.isfinite(h.twin_effect) else None,
                             range_low=round(h.twin_ci[0], 3) if np.isfinite(h.twin_ci[0]) else None,
                             range_high=round(h.twin_ci[1], 3) if np.isfinite(h.twin_ci[1]) else None,
                             edit_pairs=h.n_pairs,
                             passed_screening=h.twin_pass,
                             pooled_feature=h.feature if h.twin_pass else ""))
    return pd.DataFrame(rows)


def batteries_table(sr: ScreenResult) -> pd.DataFrame:
    df = pd.DataFrame(sr.batteries)
    if len(df):
        df["requested_by"] = df["requested_by"].apply(lambda v: ", ".join(v))
    return df


def synthetic_to_frames(world) -> tuple:
    """Export a synthetic panel in the upload format (applications, ratings, evaluators), for
    trying the upload path or as a template."""
    apps = pd.DataFrame([dict(application_id=f"A{a.app_id:03d}", text=a.text,
                              founder_gender=a.attrs["gender"]) for a in world.apps])
    ratings = pd.DataFrame([dict(evaluator_id=f"E{ev.eid:02d}", application_id=f"A{a.app_id:03d}",
                                 score=round(float(world.scores[ev.eid, a.app_id]), 2))
                            for ev in world.panel for a in world.apps])
    evaluators = pd.DataFrame([dict(evaluator_id=f"E{ev.eid:02d}", gender=ev.gender) for ev in world.panel])
    return apps, ratings, evaluators

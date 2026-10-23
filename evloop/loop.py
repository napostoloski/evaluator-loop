"""Route B: a hypothesis loop that locates what in an application drives an evaluator.

The loop runs over the whole panel in rounds. In each round:

1. Observe   For every evaluator j, the part of j's scores that is j's own: the residual
             after regressing j's scores on the leave-one-out consensus of the others (with
             rules already confirmed for j subtracted first).
2. Propose   Hypotheses for j from the text alone. A sparse model learns which text
             features predict j's residual; each hypothesis is the edit operator whose
             lexical footprint (what the edit changes in the text) best matches them.
3. Screen    Each hypothesis on a twin of j: edit every application and ask whether the
             twin's predicted residual moves, with bootstrap uncertainty.
4. Confirm   With a re-rating battery. For each feature that passed some evaluator's screen,
             a sample of applications is edited (switching the feature on where it was
             off and off where it was on), and every evaluator re-rates the edited
             versions. Each evaluator's original rating is their own control. A pooled
             model gives each evaluator's reaction to the edit. Evaluator j's rule is
             confirmed if j's reaction differs from the rest of the panel's: the test is
             idiosyncratic, not total, so reactions the whole panel shares are not reported.

Confirmed rules are subtracted, the residual is recomputed, and the loop repeats
(incremental updating). Every step is written to an audit trail.

The offline backend uses a TF-IDF lasso as the proposer and a ridge text model as the twin.
`evloop.llm` provides LLM versions of the proposer, editor, coder and twin.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge, RidgeCV

from .testbed import OPERATORS, OPS, detect, redact_gender

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Configuration and records
# ---------------------------------------------------------------------------
@dataclass
class LoopConfig:
    confirm: str = "panel"          # "none" (screen only), "twin", "panel" (full loop),
                                    # or "total" (ablation: battery tests total, not idiosyncratic, reaction)
    max_rounds: int = 2
    max_hypotheses: int = 4         # proposals per evaluator per round
    stability_B: int = 25           # lasso refits on 75% subsamples
    stability_thresh: float = 0.4   # keep features selected in at least this share of refits
    stability_alpha_scale: float = 0.6  # refit penalty relative to the cross-validated one (favours recall)
    min_effect: float = 0.10        # smallest rule worth reporting (1-5 scale), all methods
    min_twin_effect: float = 0.05   # twin screen floor; ridge twins shrink effects toward zero
    twin_boot: int = 50             # bootstrap refits of the twin
    battery_features: int = 5       # features per battery round
    battery_apps: int = 30          # applications per feature; each evaluator re-rates each once
    budget: int = 150               # max re-ratings per evaluator over all rounds (5 x 30)
    coder_error: float = 0.05       # error rate of the feature coder used for updates and tests
    rtm_adjust: bool = False        # add the original residual as a covariate in the battery (see run_battery)
    het_adjust: bool = False        # allow for real spread of reactions across evaluators in the contrast
    alpha: float = 0.05
    seed: int = 0
    # which evaluators are tested on a battery feature:
    #   "proposers": only evaluators whose own hypothesis (after screening) named the feature
    #   "nominated": also evaluators whose copy predicts a reaction to a feature another evaluator proposed
    #   "all":       every evaluator, direction read from the estimate (two-sided test)
    #   "nominated_all": nomination first (it shapes which features get batteries), then every evaluator
    test_scope: str = "proposers"
    # with test_scope="all": weight of tests backed by a screened hypothesis relative to tests run
    # only because the battery data exist (weighted BH; 1.0 = plain BH over all tests)
    prior_weight: float = 1.0


@dataclass
class Hypothesis:
    feature: str
    target: int
    proposer_score: float
    top_terms: list
    round: int
    description: str = ""
    twin_effect: float = float("nan")
    twin_ci: tuple = (float("nan"), float("nan"))
    twin_pass: bool = False
    tested: bool = False
    human_effect: float = float("nan")    # confirmed reaction relative to the panel (or total)
    human_se: float = float("nan")
    human_p: float = float("nan")
    human_q: float = float("nan")         # Benjamini-Hochberg q across the round's tests
    battery_apps: int = 0
    accepted: bool = False
    effect: float = float("nan")          # effect used when the rule is subtracted
    source: str = "proposed"              # "proposed", "nominated" (by the copy) or "panel" (tested without nomination)

    @property
    def key(self):
        return (self.feature, self.target)


@dataclass
class EvaluatorResult:
    eid: int
    accepted: list = field(default_factory=list)      # list of (feature, target, effect)
    trail: list = field(default_factory=list)         # every hypothesis considered
    queries: int = 0
    observed_premium: float = float("nan")
    observed_premium_p: float = float("nan")
    twin_premium: float = float("nan")


# ---------------------------------------------------------------------------
# Text representation
# ---------------------------------------------------------------------------
_AGREE_RE = re.compile(r"\b(she|he|it) (have|are)\b|\b(they|we) (has|is)\b")


def style_features(text: str) -> np.ndarray:
    """Writing-style statistics that a bag of words cannot see.

    These were designed alongside the synthetic generator, so they are well suited to
    detecting its writing slips. With real applications an LLM proposer replaces them.
    """
    sentences = [s for s in re.split(r"(?<=[.:])\s+|\n+", text) if s.strip()]
    n = max(len(sentences), 1)
    lower = sum(1 for s in sentences if s.strip()[0].islower()) / n
    sections = [s for s in text.split("\n\n") if s.strip()]
    unterminated = sum(1 for s in sections if not s.rstrip().endswith(".")) / max(len(sections), 1)
    agree = len(_AGREE_RE.findall(text.lower()))
    digits = len(re.findall(r"\d+", text)) / 10.0
    length = len(text.split()) / 150.0
    return np.array([lower, unterminated, agree, digits, length])


STYLE_NAMES = ["style:lowercase_starts", "style:unterminated_sections", "style:agreement_errors",
               "style:numbers", "style:length"]


class TextSpace:
    """Text representation fitted on the original applications, plus every single-feature edit.

    Columns: word and bigram TF-IDF, latent topics (LSA on the TF-IDF), and style statistics,
    standardised on the original applications. Also holds a noisy feature coding of the
    original applications (the stand-in for LLM or human coding), used for subtracting
    confirmed rules, for the outcome test and for the regression baselines.
    """

    def __init__(self, apps, max_words: int = 600, n_topics: int = 20, seed: int = 0,
                 coder_error: float = 0.05):
        self.apps = apps
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_words,
                                   lowercase=True, token_pattern=r"(?u)\b[\w'\-\[\]]+\b",
                                   sublinear_tf=True)
        self.texts = [a.text for a in apps]
        T = self.vec.fit_transform(self.texts)
        self.svd = TruncatedSVD(n_components=n_topics, random_state=seed).fit(T)
        raw = self._raw(self.texts)
        self.mu = raw.mean(0)
        self.sd = raw.std(0) + 1e-9
        self.X = (raw - self.mu) / self.sd
        self.vocab = np.array(list(self.vec.get_feature_names_out())
                              + [f"topic:{k}" for k in range(n_topics)] + STYLE_NAMES)
        self.Xedit = {}
        for op in OPERATORS:
            for v in (0, 1):
                self.Xedit[(op.name, v)] = self.transform([op.set(a, v).text for a in apps])
        # lexical footprint of each edit: mean change in the text representation
        self.signature = {op.name: (self.Xedit[(op.name, 1)] - self.Xedit[(op.name, 0)]).mean(0)
                          for op in OPERATORS}
        rng = np.random.default_rng(50_000 + seed)
        self.coded = {}
        for op in OPERATORS:
            truth = np.array([detect(op.name, t) for t in self.texts], dtype=object)
            flip = rng.random(len(truth)) < coder_error
            self.coded[op.name] = np.array([(1 - v) if (f and v is not None) else v
                                            for v, f in zip(truth, flip)], dtype=object)

    def _raw(self, texts):
        T = self.vec.transform(texts)
        return np.hstack([T.toarray(), self.svd.transform(T),
                          np.vstack([style_features(t) for t in texts])])

    def transform(self, texts):
        return (self._raw(texts) - self.mu) / self.sd

    def fires(self, feature, target):
        return np.array([v == target for v in self.coded[feature]], dtype=float)


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------
def consensus_residual(scores: np.ndarray, j: int, adjust=None):
    """Residual of j's scores (minus any confirmed rules) on the others' leave-one-out mean."""
    others = np.delete(scores, j, axis=0).mean(0)
    y = scores[j] - (adjust if adjust is not None else 0.0)
    b1, b0 = np.polyfit(others, y, 1)
    return y - (b0 + b1 * others), others


def same_gender(apps, ev_gender):
    fem = np.array([a.attrs["gender"] == "F" for a in apps])
    return fem if ev_gender == "F" else ~fem


def premium(resid, same):
    res = stats.ttest_ind(resid[same], resid[~same], equal_var=False)
    return float(resid[same].mean() - resid[~same].mean()), float(res.pvalue)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def ols(X, y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    s2 = float(e @ e) / max(df, 1)
    return beta, s2 * XtX_inv, df


# ---------------------------------------------------------------------------
# Offline proposer and twin
# ---------------------------------------------------------------------------
def cv_alpha(X, y, rng, n_alphas: int = 12, folds: int = 5):
    """Five-fold cross-validated lasso penalty on a short grid below the null penalty."""
    n = len(y)
    a_max = np.max(np.abs(X.T @ (y - y.mean()))) / n
    alphas = a_max * np.logspace(-0.05, -1.5, n_alphas)
    fold = rng.permutation(n) % folds
    err = np.zeros(n_alphas)
    for k in range(folds):
        tr, te = fold != k, fold == k
        m = Lasso(alpha=alphas[0], max_iter=3000, tol=1e-3, warm_start=True)
        for i, a in enumerate(alphas):
            m.set_params(alpha=a)
            m.fit(X[tr], y[tr])
            err[i] += np.sum((y[te] - m.predict(X[te])) ** 2)
    return float(alphas[int(np.argmin(err))])


def propose_offline(space: TextSpace, resid, excluded, cfg: LoopConfig, rng, rnd: int):
    X = space.X
    n = len(resid)
    alpha = cv_alpha(X, resid, rng) * cfg.stability_alpha_scale
    freq = np.zeros(X.shape[1]); coef = np.zeros(X.shape[1])
    for _ in range(cfg.stability_B):
        idx = rng.choice(n, int(0.75 * n), replace=False)
        m = Lasso(alpha=alpha, max_iter=3000, tol=1e-3).fit(X[idx], resid[idx])
        freq += m.coef_ != 0
        coef += m.coef_
    freq /= cfg.stability_B
    w = np.where(freq >= cfg.stability_thresh, coef / cfg.stability_B, 0.0)
    if not np.any(w):
        return []
    hyps = []
    for op in OPERATORS:
        if op.name in excluded:
            continue
        sig = space.signature[op.name]
        s = float(w @ sig)
        contrib = w * sig
        top = [space.vocab[i] for i in np.argsort(-np.abs(contrib))[:5] if contrib[i] != 0]
        if abs(s) > 1e-6 and top:
            hyps.append(Hypothesis(op.name, int(s > 0), abs(s), top, rnd))
    hyps.sort(key=lambda h: -h.proposer_score)
    return hyps[:cfg.max_hypotheses]


class RidgeTwin:
    """Text-regression twin of one evaluator: predicts the evaluator's residual from text.

    A second, dense model of the same residual the proposer used. It is a cheap filter that
    decides which hypotheses are worth real queries, not an independent source of evidence.
    """

    def __init__(self, space: TextSpace, resid, cfg: LoopConfig, rng):
        self.space = space
        self.alpha = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(space.X, resid).alpha_
        n = len(resid)
        self.models = []
        for _ in range(cfg.twin_boot):
            idx = rng.choice(n, n, replace=True)
            self.models.append(Ridge(alpha=self.alpha).fit(space.X[idx], resid[idx]))

    def edit_effect(self, feature, target):
        Xt = self.space.Xedit[(feature, target)]
        Xo = self.space.Xedit[(feature, 1 - target)]
        effs = np.array([(m.predict(Xt) - m.predict(Xo)).mean() for m in self.models])
        return float(effs.mean()), (float(np.quantile(effs, 0.025)), float(np.quantile(effs, 0.975)))

    def edit_effect_texts(self, space, texts_with, texts_without):
        """Predicted effect of arbitrary edited texts (e.g. an LLM's rewrites): mean over the
        pairs of predicted deviation with the feature minus without, with the bootstrap range."""
        if len(texts_with) < 2:
            return float("nan"), (float("nan"), float("nan"))
        Xa, Xb = space.transform(texts_with), space.transform(texts_without)
        effs = np.array([(m.predict(Xa) - m.predict(Xb)).mean() for m in self.models])
        return float(effs.mean()), (float(np.quantile(effs, 0.025)), float(np.quantile(effs, 0.975)))


# ---------------------------------------------------------------------------
# Confirmation: counterbalanced edited battery
# ---------------------------------------------------------------------------
def run_battery(world, flip_fn, n_apps: int, rng, r0, used_apps=None, rtm_adjust: bool = False):
    """Re-rating battery: every evaluator re-rates n_apps applications in edited form.

    flip_fn(app) -> (edited text, direction) where direction is +1 if the edit switched the
    feature on and -1 if it switched it off, or None to skip the application. Applications
    are drawn lazily in random order until both directions have n_apps / 2, skipping any in
    `used_apps` (so that no evaluator re-rates an application twice in a run); the chosen ones
    are added to `used_apps`.

    The outcome is the change d_ij = (re-rating of the edited version) - (original rating).
    For each evaluator separately, d is regressed on an intercept (drift between occasions),
    the evaluator's original consensus residual r0_ij, and the edit direction. The residual
    term removes regression to the mean: the hypothesis was chosen because j's original
    scores leaned one way on this feature, and part of that lean is noise that a re-rating
    would not repeat. Under the null this keeps the test unbiased; true effects are slightly
    attenuated. Returns gamma (reaction per evaluator), its variance, residual degrees of
    freedom per evaluator, and the applications used.
    """
    E = len(world.panel)
    used_apps = used_apps if used_apps is not None else set()
    half = n_apps // 2
    pos, neg = [], []
    for i in rng.permutation(len(world.apps)):
        i = int(i)
        if i in used_apps or (len(pos) >= half and len(neg) >= n_apps - half):
            continue
        app = world.apps[i]
        out = flip_fn(app)
        if out is None:
            continue
        if out[1] > 0 and len(pos) < half:
            pos.append((i, out[0], out[1]))
        elif out[1] < 0 and len(neg) < n_apps - half:
            neg.append((i, out[0], out[1]))
    chosen = pos + neg
    used_apps.update(i for i, _, _ in chosen)
    gamma, var, df = np.zeros(E), np.zeros(E), np.zeros(E)
    for j in range(E):
        d = np.array([world.query_text(j, i, text) - world.scores[j, i] for i, text, _ in chosen])
        cols = [np.ones(len(chosen))]
        if rtm_adjust:
            cols.append([r0[j, i] for i, _, _ in chosen])
        cols.append([x for _, _, x in chosen])
        beta, cov, dfj = ols(np.column_stack(cols), d)
        gamma[j], var[j], df[j] = beta[-1], cov[-1, -1], dfj
    return gamma, var, df, [i for i, _, _ in chosen]


def contrast(gamma, var, j, idiosyncratic=True, het_adjust=True):
    """Evaluator j's reaction, relative to the panel when idiosyncratic=True.

    The reference is the median of the other evaluators' reactions, so a minority with their
    own rule on the same feature does not move it. The standard error adds (a) the variance
    of the median, approximated as pi/2 times the variance of the mean, and (b) the real
    spread of reactions across evaluators, estimated robustly from the others (MAD), so that
    an evaluator must stand out from ordinary differences between evaluators.
    """
    E = len(gamma)
    if not idiosyncratic:
        return float(gamma[j]), float(np.sqrt(max(var[j], 1e-12)))
    others = np.arange(E) != j
    g = gamma[others]
    ref = float(np.median(g))
    v_ref = (np.pi / 2) * float(var[others].mean()) / (E - 1)
    mad = 1.4826 * float(np.median(np.abs(g - ref)))
    tau2 = max(0.0, mad ** 2 - float(var[others].mean())) if het_adjust else 0.0
    return float(gamma[j] - ref), float(np.sqrt(max(var[j] + v_ref + tau2, 1e-12)))


def weighted_bh(pvals, weights):
    """Weighted Benjamini-Hochberg (Genovese, Roeder & Wasserman, 2006): BH applied to p/w with
    weights rescaled to mean 1, which keeps false-discovery control while favouring tests that
    had prior support."""
    w = np.asarray(weights, dtype=float)
    w = w * len(w) / w.sum()
    return np.minimum(bh(np.asarray(pvals, dtype=float) / w), 1.0)


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * n / (rank + 1))
        q[idx] = running
    return q


# ---------------------------------------------------------------------------
# The loop over the panel
# ---------------------------------------------------------------------------
class OfflineBackend:
    """Proposer, twin, editor and coder for the offline benchmark."""

    def __init__(self, space: TextSpace):
        self.space = space

    def propose(self, j, resid, excluded, cfg, rng, rnd):
        hyps = propose_offline(self.space, resid, excluded, cfg, rng, rnd)
        for h in hyps:
            h.description = OPS[h.feature].description
        return hyps

    def screen(self, j, resid, hyps, cfg, rng):
        twin = RidgeTwin(self.space, resid, cfg, rng)
        for h in hyps:
            h.twin_effect, h.twin_ci = twin.edit_effect(h.feature, h.target)

    def flip_fn(self, feature, description):
        op = OPS[feature]
        return lambda a: (op.set(a, 1 - op.value(a)).text, 1 - 2 * op.value(a))

    def fires(self, feature, description, target):
        return self.space.fires(feature, target)


def run_panel(world, backend, cfg: LoopConfig) -> dict:
    """Run the loop for every evaluator in the panel. `backend` is an OfflineBackend, a
    TextSpace (wrapped automatically), or an LLM backend from evloop.llm."""
    if isinstance(backend, TextSpace):
        backend = OfflineBackend(backend)
    rng = np.random.default_rng(cfg.seed * 1000 + 17)
    E = len(world.panel)
    res = {j: EvaluatorResult(eid=j) for j in range(E)}
    adjust = {j: np.zeros(len(world.apps)) for j in range(E)}
    excluded = {j: set() for j in range(E)}
    done = set()
    batteries, used_apps = {}, set()
    start = {j: world.queries_used.get(j, 0) for j in range(E)}
    r0 = np.vstack([consensus_residual(world.scores, j)[0] for j in range(E)])
    for j in range(E):
        res[j].observed_premium, res[j].observed_premium_p = premium(r0[j], same_gender(world.apps, world.panel[j].gender))

    def spent():
        return max(world.queries_used.get(k, 0) - start[k] for k in range(E))

    rounds = 1 if cfg.confirm == "none" else cfg.max_rounds
    for rnd in range(1, rounds + 1):
        cands, new_rules, resids = {}, {}, {}
        for j in range(E):
            if j in done:
                continue
            ev_rng = np.random.default_rng(cfg.seed * 1000 + 100 * rnd + j)
            resid, _ = consensus_residual(world.scores, j, adjust[j])
            resids[j] = resid
            hyps = backend.propose(j, resid, excluded[j], cfg, ev_rng, rnd)
            if not hyps:
                done.add(j)
                continue
            if cfg.confirm != "none":
                backend.screen(j, resid, hyps, cfg, ev_rng)
                for h in hyps:
                    h.twin_pass = bool(h.twin_ci[0] > 0 and h.twin_effect >= cfg.min_twin_effect)
            res[j].trail.extend(hyps)
            if cfg.confirm == "none":
                new_rules[j] = [(h, h.proposer_score) for h in hyps if h.proposer_score >= cfg.min_effect]
            elif cfg.confirm == "twin":
                new_rules[j] = [(h, h.twin_effect) for h in hyps if h.twin_pass]
                excluded[j].update(h.feature for h in hyps)
            else:
                excluded[j].update(h.feature for h in hyps if not h.twin_pass)
                passing = [h for h in hyps if h.twin_pass]
                if passing:
                    cands[j] = passing

        if cfg.confirm in ("panel", "total") and cands and hasattr(backend, "canonicalize"):
            backend.canonicalize(cands)      # pool the same feature stated in different words
            cands = {j: hs for j, hs in cands.items() if hs}
        if cfg.confirm in ("panel", "total") and cands and cfg.test_scope in ("nominated", "nominated_all"):
            # screen every pooled feature for every evaluator who did not propose it; the copy's
            # prediction sets the direction, and a pass makes the evaluator a candidate
            pooled = {}
            for hs in cands.values():
                for h in hs:
                    pooled.setdefault(h.feature, h.description)
            for j in resids:
                own = {h.feature for h in cands.get(j, [])}
                todo = [f for f in pooled if f not in own and f not in excluded[j]]
                if not todo:
                    continue
                nom_rng = np.random.default_rng(cfg.seed * 1000 + 100 * rnd + j + 50_000)
                hs = [Hypothesis(f, 1, float("nan"), [], rnd, description=pooled[f], source="nominated") for f in todo]
                backend.screen(j, resids[j], hs, cfg, nom_rng)
                for h in hs:
                    if h.twin_effect < 0:
                        h.target, h.twin_effect = 0, -h.twin_effect
                        h.twin_ci = tuple(sorted((-h.twin_ci[0], -h.twin_ci[1])))
                    h.twin_pass = bool(h.twin_ci[0] > 0 and h.twin_effect >= cfg.min_twin_effect)
                passing = [h for h in hs if h.twin_pass]
                res[j].trail.extend(passing)
                excluded[j].update(h.feature for h in hs if not h.twin_pass)
                if passing:
                    cands.setdefault(j, []).extend(passing)
        if cfg.confirm in ("panel", "total") and cands:
            votes, desc = {}, {}
            for j, hs in cands.items():
                for h in hs:
                    n_v, best = votes.get(h.feature, (0, 0.0))
                    votes[h.feature] = (n_v + 1, max(best, h.twin_effect))
                    desc.setdefault(h.feature, h.description)
            # features batteried in an earlier round are tested from the stored battery
            new_feats = [f for f in sorted(votes, key=lambda f: (-votes[f][0], -votes[f][1])) if f not in batteries]
            k_max = min(cfg.battery_features, (cfg.budget - spent()) // cfg.battery_apps,
                        (len(world.apps) - len(used_apps)) // cfg.battery_apps)
            for f in new_feats[:max(k_max, 0)]:
                batteries[f] = run_battery(world, backend.flip_fn(f, desc[f]), cfg.battery_apps, rng, r0, used_apps,
                                           rtm_adjust=cfg.rtm_adjust)
            all_tested = []
            for j, hs in cands.items():
                tested = [h for h in hs if h.feature in batteries]
                for h in tested:
                    gamma, var, df, used = batteries[h.feature]
                    est, se = contrast(gamma, var, j, idiosyncratic=(cfg.confirm == "panel"), het_adjust=cfg.het_adjust)
                    sign = 1.0 if h.target == 1 else -1.0
                    h.tested, h.battery_apps = True, len(used)
                    h.human_effect, h.human_se = sign * est, se
                    h.human_p = float(2 * stats.t.sf(abs(est / se), df[j]))
                    all_tested.append((j, h))
                excluded[j].update(h.feature for h in tested)
            if cfg.test_scope in ("all", "nominated_all"):
                # test every evaluator on every battery feature they were not already tested on
                for j in range(E):
                    seen = {h.feature for h in res[j].trail if h.tested} | {f for f, _, _ in res[j].accepted}
                    for f in batteries:
                        if f in seen or f in {h.feature for _, h in all_tested if _ == j}:
                            continue
                        gamma, var, df, used = batteries[f]
                        est, se = contrast(gamma, var, j, idiosyncratic=(cfg.confirm == "panel"), het_adjust=cfg.het_adjust)
                        h = Hypothesis(f, int(est >= 0), float("nan"), [], rnd,
                                       description=desc.get(f, getattr(backend, "descriptions", {}).get(f, f)),
                                       source="panel")
                        h.tested, h.battery_apps = True, len(used)
                        h.human_effect, h.human_se = abs(est), se
                        h.human_p = float(2 * stats.t.sf(abs(est / se), df[j]))
                        res[j].trail.append(h)
                        all_tested.append((j, h))
                        excluded[j].add(f)
            if all_tested:
                # false discovery rate controlled across every test in the round, panel-wide
                ps = [h.human_p for _, h in all_tested]
                if cfg.prior_weight != 1.0 and any(h.source == "panel" for _, h in all_tested):
                    qs = weighted_bh(ps, [1.0 if h.source == "panel" else cfg.prior_weight for _, h in all_tested])
                else:
                    qs = bh(ps)
                for (j, h), q in zip(all_tested, qs):
                    h.human_q = float(q)
                    h.accepted = bool(q < cfg.alpha and h.human_effect >= cfg.min_effect)
                    if h.accepted:
                        new_rules.setdefault(j, []).append((h, h.human_effect))

        any_new = False
        for j, rules in new_rules.items():
            for h, eff in rules:
                h.accepted, h.effect = True, float(eff)
                adjust[j] = adjust[j] + h.effect * backend.fires(h.feature, h.description, h.target)
                res[j].accepted.append((h.feature, h.target, h.effect))
                any_new = True
        if not any_new and cfg.confirm in ("none", "twin"):
            break
        if cfg.confirm in ("panel", "total") and spent() + cfg.battery_apps > cfg.budget:
            break

    for j in range(E):
        res[j].queries = world.queries_used.get(j, 0) - start[j]
        same = same_gender(world.apps, world.panel[j].gender)
        pred = np.zeros(len(world.apps))
        for h in res[j].trail:
            if h.accepted:
                pred += h.effect * backend.fires(h.feature, h.description, h.target)
        res[j].twin_premium = float(pred[same].mean() - pred[~same].mean())
    return res


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
CODED = ["gender_cue", "metro_city", "consumer_sector", "solo_founder"]
LIBRARY = [op.name for op in OPERATORS]


def _robust_ols(Z, y):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ beta
    n, k = Z.shape
    bread = np.linalg.pinv(Z.T @ Z)
    V = bread @ ((Z * e[:, None] ** 2).T @ Z) @ bread * n / (n - k)
    se = np.sqrt(np.diag(V))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    return beta, p


def regression_panel(world, space: TextSpace, features=CODED, alpha=0.05, min_effect=0.10):
    """For every evaluator, regress the residual on coded features (HC1 errors); control the
    false discovery rate across all evaluator-feature tests, as the loop does. No edits."""
    tests = []
    cols = [np.array([0.5 if x is None else float(x) for x in space.coded[c]]) for c in features]
    Z = np.column_stack([np.ones(len(world.apps))] + cols)
    for j in range(len(world.panel)):
        resid, _ = consensus_residual(world.scores, j)
        beta, p = _robust_ols(Z, resid)
        tests += [(j, c, b, pp) for c, b, pp in zip(features, beta[1:], p[1:])]
    q = bh([t[3] for t in tests])
    out = {j: [] for j in range(len(world.panel))}
    for (j, c, b, _), qq in zip(tests, q):
        if qq < alpha and abs(b) >= min_effect:
            out[j].append((c, int(b > 0), float(abs(b))))
    return out


def edit_everything(world, budget: int, rng, alpha=0.05, min_effect=0.10):
    """No proposer, no twin: spend the same re-rating budget on every feature in the library.

    The budget is split evenly over the ten features (no application is re-rated twice),
    each evaluator's idiosyncratic reaction is tested in both directions, and false
    discoveries are controlled panel-wide. The natural design for a short, known list.
    """
    E = len(world.panel)
    n_apps = max(2, min(budget, len(world.apps)) // len(OPERATORS))
    r0 = np.vstack([consensus_residual(world.scores, j)[0] for j in range(E)])
    used, tests = set(), []
    for op in OPERATORS:
        gamma, var, df, _ = run_battery(
            world, lambda a, op=op: (op.set(a, 1 - op.value(a)).text, 1 - 2 * op.value(a)), n_apps, rng, r0, used)
        for j in range(E):
            est, se = contrast(gamma, var, j)
            tests.append((j, op.name, est, float(2 * stats.t.sf(abs(est / se), df[j]))))
    q = bh([t[3] for t in tests])
    out = {j: [] for j in range(E)}
    for (j, f, est, p), qq in zip(tests, q):
        if qq < alpha and abs(est) >= min_effect:
            out[j].append((f, int(est > 0), float(abs(est))))
    return out


def redaction_battery(world, rng, alpha=0.05, n_apps=None):
    """Prereg-style diagnostic as a re-rating battery. Each evaluator re-rates applications,
    a random half with gender cues redacted and half unchanged. For each evaluator, does
    redaction lower re-ratings of same-gender applications more than of other-gender ones,
    relative to unchanged re-ratings? Returns {eid: (verdict, p)}.
    """
    E, N = len(world.panel), len(world.apps)
    idx = rng.permutation(N)[: (n_apps or N)]
    fem = np.array([a.attrs["gender"] == "F" for a in world.apps])
    out = {}
    for j in range(E):
        r0, _ = consensus_residual(world.scores, j)
        red = rng.random(len(idx)) < 0.5
        same = np.array([fem[i] == (world.panel[j].gender == "F") for i in idx], dtype=float)
        d = np.array([world.query_text(j, int(i), redact_gender(world.apps[i]).text if r else world.apps[i].text)
                      - world.scores[j, i] for i, r in zip(idx, red)])
        X = np.column_stack([np.ones(len(idx)), r0[idx], same, red, red * same])
        beta, cov, df = ols(X, d)
        t = beta[4] / np.sqrt(cov[4, 4])
        p = float(2 * stats.t.sf(abs(t), df))
        out[j] = ("cue" if (p < alpha and beta[4] < 0) else "not cue", p)
    return out


def outcome_label(world, space: TextSpace, j: int, feature, target, alpha=0.05):
    """Ability or bias: does the rewarded feature predict outcomes beyond the consensus score?"""
    f = space.fires(feature, target)
    cons = np.delete(world.scores, j, axis=0).mean(0)
    Z = np.column_stack([np.ones_like(cons), cons, f])
    beta, cov, df = ols(Z, world.outcome)
    p = 2 * stats.t.sf(abs(beta[2] / np.sqrt(cov[2, 2])), df)
    return "ability" if (p < alpha and beta[2] > 0) else "bias"


def trail_records(res: EvaluatorResult):
    rows = []
    for h in res.trail:
        d = asdict(h)
        d["eid"] = res.eid
        d["twin_ci_lo"], d["twin_ci_hi"] = d.pop("twin_ci")
        d["top_terms"] = " | ".join(d["top_terms"])
        rows.append(d)
    return rows

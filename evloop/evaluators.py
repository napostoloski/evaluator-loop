"""Synthetic evaluators with planted rules, synthetic outcomes, and a re-query oracle.

Each evaluator reads an application's text. Its score combines a noisy perception of the
application's latent quality with any planted rules. A rule fires when a feature, as read
from the text by `testbed.detect`, takes a target value. Because rules act on the text,
redacting or swapping a cue changes the score exactly as it would for a reader who reacts
to that cue.

The panel mirrors the real study: 20 evaluators, 10 women and 10 men, each scoring every
application on five 1-5 dimensions; the overall score is their mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .testbed import Application, detect

DIMENSIONS = ["problem_solution", "uniqueness", "business_model", "growth", "team"]


@dataclass(frozen=True)
class Rule:
    feature: str      # operator / detector name
    target: int       # feature value that earns the premium
    delta: float      # premium on the 1-5 scale

    @property
    def key(self):
        return (self.feature, self.target)


# Reactions every evaluator shares. They make edits move everyone's scores, so the loop has
# to separate what is particular to one evaluator from what the whole panel does.
SHARED_REACTIONS = [Rule("traction", 1, 0.30), Rule("fluency_high", 1, 0.15),
                    Rule("experienced", 1, 0.15), Rule("solo_founder", 1, -0.15)]


@dataclass
class Evaluator:
    eid: int
    gender: str
    role: str
    leniency: float
    skill: float
    perception_sd: float
    rules: list = field(default_factory=list)        # planted, idiosyncratic rules
    shared: list = field(default_factory=lambda: list(SHARED_REACTIONS))
    dim_sd: float = 0.35
    retest_sd: float = 0.15

    def latent(self, text: str, quality: float, perception: float) -> float:
        x = 3.0 + self.leniency + self.skill * (quality + perception)
        for r in self.shared + self.rules:
            if detect(r.feature, text) == r.target:
                x += r.delta
        return x

    def score_text(self, text: str, quality: float, perception: float, rng: np.random.Generator,
                   retest: bool = False) -> float:
        x = self.latent(text, quality, perception)
        if retest:
            x += rng.normal(0, self.retest_sd)
        dims = np.clip(np.round(x + rng.normal(0, self.dim_sd, len(DIMENSIONS))), 1, 5)
        return float(dims.mean())

    def score(self, app: Application, perception: float, rng: np.random.Generator,
              retest: bool = False) -> float:
        return self.score_text(app.text, app.attrs["quality"], perception, rng, retest)


ROLE_ORDER = ["cue", "cue", "proxy", "proxy_home", "fluency", "ability", "multi", "null", "null", "null"]


def make_panel(seed: int = 0, delta: float = 0.5) -> list[Evaluator]:
    """10 female and 10 male evaluators with planted roles.

    cue:      same-gender premium triggered by the founder's name and pronouns (projection via cue)
    proxy:    premium for a sector cluster that is correlated with founder gender
              (women evaluators favour consumer sectors, men favour technology)
    proxy_home: premium tied to running the business from home, which is also correlated
              with founder gender but is not a variable in the usual structured data
              (women evaluators favour home-based ventures, men penalise them)
    fluency:  premium for fluent English (correlated with large-city founders)
    ability:  premium for reported paying customers, a feature that predicts outcomes
    multi:    cue rule plus ability rule
    null:     no planted rule

    On top of its planted rules, every evaluator shares the reactions in SHARED_REACTIONS.
    Planted rules are what the loop must find; shared reactions it must not report.
    """
    rng = np.random.default_rng(10_000 + seed)
    panel = []
    for k in range(20):
        g = "F" if k < 10 else "M"
        role = ROLE_ORDER[k % 10]
        own = 1 if g == "F" else 0
        rules = []
        if role in ("cue", "multi"):
            rules.append(Rule("gender_cue", own, delta))
        if role == "proxy":
            rules.append(Rule("consumer_sector", own, delta))
        if role == "proxy_home":
            rules.append(Rule("home_based", own, delta))
        if role == "fluency":
            rules.append(Rule("fluency_high", 1, delta))
        if role in ("ability", "multi"):
            rules.append(Rule("traction", 1, delta))
        # each evaluator's shared reactions vary a little around the panel values
        shared = [Rule(r.feature, r.target, float(r.delta + rng.normal(0, 0.1))) for r in SHARED_REACTIONS]
        panel.append(Evaluator(
            eid=k, gender=g, role=role,
            leniency=float(rng.normal(0, 0.3)),
            skill=float(rng.uniform(0.35, 0.7)),
            perception_sd=float(rng.uniform(1.0, 1.6)),
            rules=rules,
            shared=shared,
        ))
    return panel


@dataclass
class World:
    """One synthetic replication: applications, panel, observed scores, outcomes."""
    apps: list
    panel: list
    perception: np.ndarray          # evaluator x application perception noise (fixed per pair)
    scores: np.ndarray              # evaluator x application observed overall scores
    outcome: np.ndarray             # application log-revenue outcome
    rng: np.random.Generator
    rho: float = 0.6                # correlation of an evaluator's reading of a venture across occasions
    queries_used: dict = field(default_factory=dict)

    def query_text(self, eid: int, app_id: int, text: str) -> float:
        """Ask evaluator `eid` to score a version of application `app_id` given as text.

        Mirrors asking a human evaluator to rate an edited application months after the first
        rating: same evaluator and venture, but the evaluator's reading of the venture is
        correlated rho with their first reading, plus fresh rating noise. The evaluator reads
        the text, so an edit counts only if it changes what the evaluator perceives. Every
        call is counted against the evaluator's budget.
        """
        self.queries_used[eid] = self.queries_used.get(eid, 0) + 1
        ev = self.panel[eid]
        q = self.apps[app_id].attrs["quality"]
        # the evaluator's reading of the venture is only partly the same as on the first occasion
        p = self.rho * self.perception[eid, app_id] + np.sqrt(1 - self.rho ** 2) * self.rng.normal(0, ev.perception_sd)
        return ev.score_text(text, q, p, self.rng, retest=True)

    def query(self, eid: int, app: Application) -> float:
        return self.query_text(eid, app.app_id, app.text)


def make_world(apps: list, seed: int = 0, delta: float = 0.5, rho: float = 0.6) -> World:
    rng = np.random.default_rng(20_000 + seed)
    panel = make_panel(seed, delta)
    n = len(apps)
    perception = np.vstack([rng.normal(0, ev.perception_sd, n) for ev in panel])
    scores = np.array([[ev.score(a, perception[ev.eid, a.app_id], rng) for a in apps] for ev in panel])
    q = np.array([a.attrs["quality"] for a in apps])
    traction = np.array([float(a.attrs["traction"]) for a in apps])
    stage = np.array([a.attrs["stage"] for a in apps])
    outcome = 0.8 * q + 0.5 * traction + 0.2 * stage + rng.normal(0, 1.5, n)
    return World(apps, panel, perception, scores, outcome, rng, rho=rho)

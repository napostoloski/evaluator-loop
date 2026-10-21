import numpy as np

from evloop.evaluators import make_world
from evloop.loop import CODED, LIBRARY, LoopConfig, TextSpace, bh, holm, regression_panel, run_panel
from evloop.testbed import OPS, sample_applications


def test_planted_rule_moves_scores():
    apps = sample_applications(156, seed=2)
    w = make_world(apps, seed=2, delta=0.75)
    ev = w.panel[5]                       # woman, ability role: rewards paying customers
    assert ev.role == "ability"
    op = OPS["traction"]
    d = [w.query(5, op.set(a, 1)) - w.query(5, op.set(a, 0)) for a in apps[:60]]
    assert abs(np.mean(d) - 0.75) < 0.25


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.all(adj <= 1) and adj[0] == 0.04 and adj[3] == 0.5


def test_bh_matches_definition():
    q = bh([0.01, 0.04, 0.03, 0.5])
    assert abs(q[0] - 0.04) < 1e-12 and abs(q[1] - 0.0533333) < 1e-6 and q[3] == 0.5


def test_full_loop_recovers_strong_rules_and_ignores_shared_reactions():
    apps = sample_applications(156, seed=7)
    w = make_world(apps, seed=7, delta=0.75)
    space = TextSpace(apps, seed=7)
    cfg = LoopConfig(seed=7)
    res = run_panel(w, space, cfg)
    tp = fp = fn = 0
    for j, r in res.items():
        planted = {x.key for x in w.panel[j].rules}
        got = {(f, t) for f, t, _ in r.accepted}
        tp, fp, fn = tp + len(got & planted), fp + len(got - planted), fn + len(planted - got)
        assert r.queries <= cfg.budget
    assert tp >= 13 and fp <= 2, (tp, fp, fn)
    # the shared reaction to paying customers is never reported for a null evaluator
    for j in (7, 8, 9, 17, 18, 19):
        assert ("traction", 1) not in {(f, t) for f, t, _ in res[j].accepted}


def test_coded_regression_cannot_see_uncoded_rules():
    apps = sample_applications(156, seed=7)
    w = make_world(apps, seed=7, delta=0.75)
    space = TextSpace(apps, seed=7)
    acc = regression_panel(w, space, CODED)[5]    # traction is not a coded covariate
    assert not any(f == "traction" for f, _, _ in acc)
    assert any(f == "traction" for f, _, _ in regression_panel(w, space, LIBRARY)[5])

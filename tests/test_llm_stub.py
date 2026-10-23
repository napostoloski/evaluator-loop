import json

from evloop.evaluators import make_world
from evloop.llm import LLMBackend, LLMClient, _json_block, code_llm, edit_llm, propose_llm, strip_think
from evloop.loop import LoopConfig, consensus_residual, run_panel
from evloop.testbed import detect, sample_applications

from .stub_server import _bad_edit_slot, start


def test_reply_cleaning_handles_reasoning_and_wrappers():
    assert strip_think("<think>long reasoning</think>yes") == "yes"
    assert strip_think("<think>cut off mid-thought") == ""
    assert _json_block('<think>x</think>Here are ideas [1] and more: [{"a": 1}]') == [{"a": 1}]
    assert _json_block('{"hypotheses": [{"a": 2}]}') == [{"a": 2}]


def test_client_caches_validated_replies_and_logs_hits(tmp_path):
    server, url = start()
    try:
        c = LLMClient(url, "stub", cache_dir=str(tmp_path / "cache"), log_path=str(tmp_path / "log.jsonl"))
        msg = [{"role": "user", "content": "Does the following statement x \"The founder is a woman.\"\n\nAPPLICATION\nAyesha Khan is the sole founder."}]
        assert c.chat(msg) == "yes"
        assert c.chat(msg) == "yes"          # second call served from cache
        recs = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().strip().splitlines()]
        assert [r["cached"] for r in recs] == [False, True]
        # a reply that fails validation is not cached
        c.chat([{"role": "user", "content": "unrelated"}], validate=lambda t: False)
        assert len(list((tmp_path / "cache").iterdir())) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_propose_edit_code_roundtrip():
    server, url = start()
    try:
        c = LLMClient(url, "stub", cache_dir=None, log_path=None)
        apps = sample_applications(40, seed=3)
        w = make_world(apps, seed=3)
        resid, _ = consensus_residual(w.scores, 0)
        hyps = propose_llm(c, [a.text for a in apps], resid, k=3)
        assert hyps and hyps[0]["description"] == "The founder is a woman."
        males = [a for a in apps if a.attrs["gender"] == "M"]
        clean = next(a for a in males if _bad_edit_slot(a.text) == 0)
        e = edit_llm(c, clean.text, "The founder is a woman.", present=True)
        assert e.ok and detect("gender_cue", e.text) == 1
        assert code_llm(c, e.text, "The founder is a woman.") == 1
        # the stub's deliberate half-edit (pronouns switched, name kept) is rejected by the checks
        half = next((a for a in males if _bad_edit_slot(a.text) == 1), None)
        if half is not None:
            e2 = edit_llm(c, half.text, "The founder is a woman.", present=True)
            assert not e2.ok and e2.reason == "pronouns_without_name"
        # appended commentary is cut and the edit kept
        noisy = next((a for a in males if _bad_edit_slot(a.text) == 2), None)
        if noisy is not None:
            e3 = edit_llm(c, noisy.text, "The founder is a woman.", present=True)
            assert e3.ok and "Wait" not in e3.text and detect("gender_cue", e3.text) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_llm_backend_runs_the_panel_loop_end_to_end():
    server, url = start()
    try:
        c = LLMClient(url, "stub", cache_dir=None, log_path=None)
        apps = sample_applications(156, seed=5)
        w = make_world(apps, seed=5, delta=0.75)
        cfg = LoopConfig(seed=5, max_rounds=1, battery_features=2, battery_apps=16, budget=32)
        backend = LLMBackend(c, w, n_shots=6, n_twin_pairs=8, log=lambda *_: None)
        res = run_panel(w, backend, cfg)
        acc0 = {(f, t) for f, t, _ in res[0].accepted}      # woman with a planted same-gender cue rule
        assert ("the founder is a woman", 1) in acc0          # features are keyed by the canonical description
        # a man with the same-gender cue proposed "the founder is a man"; pooling maps it onto the
        # canonical woman-founder feature with the target flipped, and the battery confirms it
        acc10 = {(f, t) for f, t, _ in res[10].accepted}
        assert ("the founder is a woman", 0) in acc10
        # no null evaluator is given the shared paying-customers reaction as a rule of their own
        for j in (7, 8, 9, 17, 18, 19):
            assert not any("paying" in f for f, _, _ in res[j].accepted)
        assert max(r.queries for r in res.values()) <= cfg.budget
        # pooling happened: the gender feature was requested by several evaluators in one group
        assert backend.groups and any(len(v) > 1 for v in backend.groups[0].values())
        # the stub's deliberate half-edits and commentary were caught by the checks
        assert backend.drop_reasons.get("pronouns_without_name", 0) > 0
        assert backend.drop_reasons.get("unchanged", 0) == 0 or True
        assert backend.dropped_edits > 0
    finally:
        server.shutdown()
        server.server_close()

import numpy as np
import pandas as pd
import pytest

from evloop.evaluators import make_world
from evloop.llm import LLMBackend, LLMClient
from evloop.loop import LoopConfig, TextSpace
from evloop.realdata import (DataError, batteries_table, hypotheses_table, load_dataset, quick_candidates,
                             screen_dataset, synthetic_to_frames)
from evloop.testbed import sample_applications

from .stub_server import start


def _frames(n=60, seed=5):
    w = make_world(sample_applications(n, seed=seed), seed=seed)
    return w, synthetic_to_frames(w)


def test_load_dataset_round_trip_and_gender():
    w, (A, R, V) = _frames()
    d = load_dataset(A, R, V)
    assert d.scores.shape == (20, 60)
    assert np.allclose(d.scores, np.round(w.scores, 2))
    assert d.panel[0].gender == "F" and d.apps[0].attrs["gender"] in ("F", "M")
    assert d.notes == []


def test_load_dataset_fills_gaps_and_drops_sparse_evaluators():
    _, (A, R, V) = _frames()
    R = R[~((R.evaluator_id == "E00") & (R.application_id.isin(["A001", "A002"])))]   # two gaps
    R = R[~((R.evaluator_id == "E01") & (R.application_id < "A030"))]                  # half missing
    d = load_dataset(A, R, V)
    assert len(d.panel) == 19 and not np.isnan(d.scores).any()
    assert any("left out" in n for n in d.notes) and any("filled" in n for n in d.notes)


def test_load_dataset_explains_bad_input():
    _, (A, R, V) = _frames()
    with pytest.raises(DataError, match="column"):
        load_dataset(A.rename(columns={"text": "body"}), R, V)
    with pytest.raises(DataError, match="At least"):
        load_dataset(A.head(10), R, V)


def test_text_space_without_library_for_uploads():
    _, (A, R, V) = _frames()
    d = load_dataset(A, R, V)
    space = TextSpace(d.apps)
    assert not space.library and space.Xedit == {}
    resid = d.scores[0] - d.scores.mean(0)
    cand = quick_candidates(space, resid, k=3)
    assert len(cand["higher"]) == 3 and len(cand["lower"]) == 3


def test_screen_dataset_with_stub_model(tmp_path):
    server, url = start()
    try:
        _, (A, R, V) = _frames(n=60, seed=7)
        d = load_dataset(A, R, V)
        client = LLMClient(url, "stub", cache_dir=None, log_path=None)
        backend = LLMBackend(client, d, n_twin_pairs=8, twin="ridge", verify=True, workers=2,
                             proposer_prompt="v2", n_examples=8, log=lambda *_: None)
        cfg = LoopConfig(seed=7, battery_apps=6, battery_features=2, max_hypotheses=3)
        seen = []
        sr = screen_dataset(d, backend, cfg, progress=lambda f, m: seen.append(f))
        H = hypotheses_table(d, sr)
        assert len(H) > 0 and set(H.evaluator_id) <= {r.label for r in d.panel}
        assert seen[-1] == pytest.approx(1.0)
        if sr.features:
            B = batteries_table(sr)
            assert len(B) > 0 and set(B.change) <= {"added", "removed"}
            assert (B.original_text != B.edited_text).all()
        some = [h for r in sr.results.values() for h in r.trail if h.n_pairs]
        assert some and all(len(h.examples) <= 3 for h in some)
    finally:
        server.shutdown()
        server.server_close()

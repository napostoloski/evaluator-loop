import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "../evloop/app.py"


def test_simulated_page_runs_offline_and_shows_results():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Simulated panel" in h.value for h in at.header)
    at.button[0].click().run()          # "Run the prototype" with the default (offline) engine
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Planted preferences found" in labels
    assert len(at.dataframe) >= 2       # overview and one evaluator's hypotheses


def test_upload_page_loads_without_files():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Your own data").run()
    assert not at.exception
    assert any("Your own data" in h.value for h in at.header)

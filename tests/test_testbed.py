import numpy as np

from evloop.testbed import OPERATORS, OPS, detect, redact_geo, redact_gender, render_sections, sample_applications


def test_rendering_is_deterministic():
    a = sample_applications(10, seed=4)
    b = sample_applications(10, seed=4)
    assert [x.text for x in a] == [x.text for x in b]


def test_detectors_read_every_edit_correctly():
    for app in sample_applications(60, seed=11):
        for op in OPERATORS:
            for v in (0, 1):
                e = op.set(app, v)
                assert detect(op.name, e.text) == op.value(e), (op.name, v, e.text)


def test_edits_are_minimal():
    """A single-feature edit leaves untouched sections word-for-word identical."""
    owner = {"traction": {"Problem to solve"}, "home_based": {"Team"}, "foreign_degree": {"Team"},
             "family_business": {"Team"}, "solo_founder": {"Team"}, "experienced": {"Team"},
             "gender_cue": {"Team"}}
    for app in sample_applications(40, seed=12):
        base = render_sections(app)
        for name, sections in owner.items():
            op = OPS[name]
            edited = render_sections(op.set(app, 1 - op.value(app)))
            for k in base:
                if k not in sections:
                    assert base[k] == edited[k], (name, k)


def test_redaction_removes_cues():
    for app in sample_applications(30, seed=13):
        assert detect("gender_cue", redact_gender(app).text) is None
        assert detect("metro_city", redact_geo(app).text) is None
        assert "[Founder]" in redact_gender(app).text


def test_built_in_confounds_exist():
    apps = sample_applications(2000, seed=0)
    fem = np.array([a.attrs["gender"] == "F" for a in apps])
    consumer = np.array([a.attrs["cluster"] == "consumer" for a in apps])
    home = np.array([a.attrs["home_based"] for a in apps])
    assert consumer[fem].mean() - consumer[~fem].mean() > 0.25
    assert home[fem].mean() - home[~fem].mean() > 0.15


def test_edits_change_no_other_feature():
    for app in sample_applications(80, seed=14):
        for op in OPERATORS:
            e = op.set(app, 1 - op.value(app))
            for other in OPERATORS:
                if other.name != op.name:
                    assert detect(other.name, e.text) == detect(other.name, app.text), (op.name, other.name)

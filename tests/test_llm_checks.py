"""Checks on the LLM path that need no model: rewrite cleaning, edit sanity, canonicalisation.
The example strings are taken from the first real-model run (Claude Sonnet 5.5, 23 Oct 2026)."""
from evloop.llm import (LLMClient, _clean_rewrite, canonicalize_llm, canonicalize_offline, edit_sanity,
                        word_similarity)

from .stub_server import start

ORIGINAL = ("Business idea: we are a Peshawar-based startup developing a cloud-based inventory app for small retailers.\n\n"
            "Problem to solve: small retailers track stock on paper and lose sales when items run out.\n\n"
            "Target customer: We serve small retail shops and pharmacies, starting in Peshawar.\n\n"
            "Team: venture is led by Kamran Farooqi, working with two co-founders. He has 10 years of experience "
            "in software development. He has a bachelor's degree from a local university.")


def test_half_edit_pronouns_without_name_is_rejected():
    half = ORIGINAL.replace("He has", "She has")
    ok, reason = edit_sanity(ORIGINAL, half)
    assert not ok and reason == "pronouns_without_name"


def test_full_gender_swap_is_accepted():
    full = ORIGINAL.replace("Kamran", "Kamila").replace("He has", "She has")
    assert edit_sanity(ORIGINAL, full) == (True, "")


def test_mixed_pronouns_are_rejected():
    mixed = ORIGINAL.replace("Kamran", "Kamila").replace("He has 10", "She has 10")
    ok, reason = edit_sanity(ORIGINAL, mixed)
    assert not ok and reason == "mixed_pronouns"


def test_trailing_commentary_is_cut_and_the_rest_kept():
    reply = ORIGINAL.replace("Kamran", "Kamila").replace("He has", "She has") + \
        "  Wait — the name must also change, so I used Kamila."
    cleaned = _clean_rewrite(reply, ORIGINAL)
    assert "Wait" not in cleaned and cleaned.endswith("local university.")
    assert edit_sanity(ORIGINAL, cleaned)[0]


def test_preamble_and_leading_commentary_are_removed():
    reply = "The statement is already FALSE: the founder has a male name. Here is the application unchanged.\n\n" + ORIGINAL
    assert _clean_rewrite(reply, ORIGINAL) == ORIGINAL


def test_rewrite_that_drops_a_section_is_rejected():
    truncated = ORIGINAL.split("\n\nTeam:")[0]
    ok, reason = edit_sanity(ORIGINAL, truncated)
    assert not ok and reason == "sections"
    assert word_similarity(ORIGINAL, truncated) < 1


def test_offline_canonicaliser_groups_rewordings():
    descs = ["The application states that the business currently runs from the founder's home.",
             "The team section states that the business currently runs from the founder's home.",
             "The founder holds a master's degree from a university in the United Kingdom.",
             "The founder is described as holding a master's degree from a UK university rather than a local bachelor's.",
             "The application describes a founder leading with two or more co-founders rather than being a sole founder."]
    out = canonicalize_offline(descs)
    assert out[0][0] == out[1][0]            # the two home-based wordings share a canonical statement
    assert out[2][0] == out[3][0]            # the two UK-degree wordings too
    assert out[4][0] not in (out[0][0], out[2][0])
    assert not any(flip for _, flip in out)  # the offline fallback never infers polarity


def test_llm_canonicaliser_merges_opposites_with_a_flip():
    server, url = start()
    try:
        c = LLMClient(url, "stub", cache_dir=None, log_path=None)
        out = canonicalize_llm(c, ["The founder is a woman.", "The founder is a man.",
                                   "The application reports paying pilot customers."])
        assert out[0] == ("The founder is a woman.", False)
        assert out[1] == ("The founder is a woman.", True)
        assert out[2][0].startswith("The application reports paying") and out[2][1] is False
    finally:
        server.shutdown()
        server.server_close()

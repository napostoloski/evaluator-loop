"""Local app for the prototype.

    pip install -e ".[app]"
    python -m evloop app            # opens http://localhost:8501

Two modes:
  Simulated panel: choose the settings, see how the synthetic applications and evaluators are
      made, run the prototype (offline in about 20 seconds, or with an AI model), and inspect
      every evaluator's tested hypotheses against the planted truth.
  Your own data:  upload applications and ratings; an AI model proposes and screens hypotheses
      for each evaluator and prepares the edited applications a confirmation study would use.
      Nothing is confirmed without re-ratings.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from evloop.present import (HOW_APPLICATIONS, HOW_EVALUATORS, HOW_PROTOTYPE, SETTINGS, UPLOAD_FORMAT,
                            simulated_summary, upload_summary)

RUN_DIR = os.environ.get("EVLOOP_RUN_DIR", os.path.join(os.getcwd(), "evloop_runs"))

PROVIDERS = {
    "Anthropic (Claude)": dict(base_url="https://api.anthropic.com/v1", model="claude-sonnet-5-5",
                               extra_body='{"thinking": {"type": "between_tools"}}', max_tokens_field="max_tokens"),
    "OpenAI": dict(base_url="https://api.openai.com/v1", model="", extra_body="",
                   max_tokens_field="max_completion_tokens"),
    "Other (OpenAI-compatible, e.g. a local open-weight model)": dict(
        base_url="http://localhost:8000/v1", model="", extra_body="", max_tokens_field="max_tokens"),
}

st.set_page_config(page_title="Evaluator loop prototype", layout="wide")


# ---------------------------------------------------------------------------
# AI model settings (sidebar)
# ---------------------------------------------------------------------------
def model_settings():
    with st.sidebar.expander("AI model", expanded=False):
        st.caption("Needed for your own data, optional for the simulated panel. Everything sent to the model "
                   "goes to the provider you choose, so check that your data permissions allow it.")
        prov = st.selectbox("Provider", list(PROVIDERS), key="provider")
        d = PROVIDERS[prov]
        base_url = st.text_input("API address", d["base_url"], key=f"url_{prov}")
        model = st.text_input("Model", d["model"], key=f"model_{prov}", placeholder="model name")
        key = st.text_input("API key", type="password", key=f"key_{prov}",
                            help="Kept only in this session. You can also set EVLOOP_API_KEY before starting the app.")
        extra = st.text_input("Extra request settings (JSON)", d["extra_body"], key=f"extra_{prov}",
                              help="For Claude, this turns off extended thinking so short answers are not cut off.")
        workers = st.slider("Parallel requests", 1, 8, 4, key="workers")
    return dict(base_url=base_url, model=model, api_key=key or os.environ.get("EVLOOP_API_KEY"),
                extra_body=extra, workers=workers, max_tokens_field=d["max_tokens_field"])


def make_client(ms, run_name):
    from evloop.llm import LLMClient
    out = os.path.join(RUN_DIR, run_name)
    os.makedirs(out, exist_ok=True)
    extra = json.loads(ms["extra_body"]) if ms["extra_body"].strip() else {}
    return LLMClient(ms["base_url"], ms["model"], api_key=ms["api_key"], temperature=None, extra_body=extra,
                     max_tokens_field=ms["max_tokens_field"], cache_dir=os.path.join(out, "llm_cache"),
                     log_path=os.path.join(out, "llm_calls.jsonl")), out


def model_ready(ms):
    if not ms["model"] or not ms["base_url"]:
        st.warning("Choose a provider and model under **AI model** in the sidebar.")
        return False
    return True


# ---------------------------------------------------------------------------
# Results: simulated panel
# ---------------------------------------------------------------------------
def show_simulated(summary):
    s = summary["summary"]
    st.subheader("Results")
    c = st.columns(4)
    c[0].metric("Planted preferences found", f"{s['found']} of {s['planted']}")
    c[1].metric("Reported preferences that were real",
                "–" if s["real_share"] is None else f"{100 * s['real_share']:.0f}%",
                help=f"{s['real']} of {s['reported']} confirmed preferences match a planted one.")
    c[2].metric("Evaluators without a preference wrongly flagged", f"{s['null_flagged']} of {s['null_evaluators']}")
    c[3].metric("Re-ratings per evaluator", max(e["rerated"] for e in summary["evaluators"]))

    over = pd.DataFrame([dict(Evaluator=e["id"], Gender=e["gender"], **{"Planted (the truth)": "; ".join(
        p["text"] + (" ✓" if p["found"] else " ✗") for p in e["planted"]) or "none"},
        **{"Confirmed by the prototype": "; ".join(e["confirmed"]) or "none",
           "Hypotheses considered": len(e["hypotheses"])}) for e in summary["evaluators"]])
    st.markdown("**All evaluators.** ✓ found, ✗ missed.")
    st.dataframe(over, hide_index=True)

    st.markdown("**One evaluator in detail**")
    ids = [e["id"] for e in summary["evaluators"]]
    pick = st.selectbox("Evaluator", ids, format_func=lambda i: f"Evaluator {i} ({summary['evaluators'][i]['gender']}): "
                                                                f"{summary['evaluators'][i]['role_text']}")
    e = summary["evaluators"][pick]
    st.markdown("Planted: " + ("; ".join(p["text"] + (" (found)" if p["found"] else " (missed)") for p in e["planted"])
                               or "no planted preference"))
    rows = []
    for h in e["hypotheses"]:
        rows.append({"Hypothesis": h["text"], "Direction": h["direction"], "Round": h["round"],
                     "Copy's predicted effect": h["screen_effect"],
                     "Range": None if h["screen_low"] is None else f"{h['screen_low']:+.2f} to {h['screen_high']:+.2f}",
                     "Re-rating effect vs panel": h["effect"], "q": h["q"],
                     "Outcome": h["verdict"], "Planted?": "yes" if h["planted"] else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Copy's predicted effect: how much the statistical copy of this evaluator expects the edit to move "
               "their score (points), with the range across 50 refits. Re-rating effect: how much more this evaluator "
               "reacted to the edit than the rest of the panel did. q: the p-value corrected for the number of tests; "
               "a preference is confirmed when q < 0.05 and the effect is at least 0.10.")
    st.download_button("Download the full results (JSON)", json.dumps(summary, indent=1), "prototype_run.json",
                       "application/json")


# ---------------------------------------------------------------------------
# Results: uploaded data
# ---------------------------------------------------------------------------
def show_upload(summary, H, B):
    st.subheader("Screened hypotheses")
    st.info("These are hypotheses that passed a cheap screen, not findings. Each would need re-ratings by the "
            "evaluator to be confirmed.")
    if summary["shared_features"]:
        st.markdown("**Hypotheses that passed screening, and for which evaluators**")
        st.dataframe(pd.DataFrame([dict(Hypothesis=f["text"], Evaluators=", ".join(f["evaluators"]))
                                   for f in summary["shared_features"]]), hide_index=True)
    ids = [e["id"] for e in summary["evaluators"]]
    pick = st.selectbox("Evaluator", range(len(ids)), format_func=lambda i: f"Evaluator {ids[i]}")
    e = summary["evaluators"][pick]
    if e["premium"] is not None:
        st.markdown(f"Same-gender premium in the original ratings: **{e['premium']:+.2f}** points (p = {e['premium_p']}).")
    rows = [{"Hypothesis": h["text"], "Direction": h["direction"], "Copy's predicted effect": h["screen_effect"],
             "Range": None if h["screen_low"] is None else f"{h['screen_low']:+.2f} to {h['screen_high']:+.2f}",
             "Edit pairs": h["pairs"], "Outcome": h["verdict"]} for h in e["hypotheses"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for h in e["hypotheses"]:
        if h["examples"]:
            with st.expander(f"Example edit: {h['text'][:90]}"):
                ex = h["examples"][0]
                c = st.columns(2)
                c[0].markdown("**With the feature**")
                c[0].text(ex["with_feature"])
                c[1].markdown("**Without the feature**")
                c[1].text(ex["without_feature"])
    st.download_button("Download all hypotheses (CSV)", H.to_csv(index=False), "hypotheses.csv", "text/csv")
    if B is not None and len(B):
        st.download_button(f"Download the {len(B)} edited applications to re-rate (CSV)", B.to_csv(index=False),
                           "edits_to_rerate.csv", "text/csv")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_simulated(ms):
    st.header("Simulated panel")
    st.write("Choose the settings, run the prototype, and compare what it finds with the preferences planted in "
             "the simulated evaluators.")
    with st.expander("How the synthetic applications are created"):
        st.write(HOW_APPLICATIONS)
        from evloop.testbed import sample_applications
        st.markdown("**An example application**")
        st.text(sample_applications(3, seed=1)[0].text)
    with st.expander("How the simulated evaluators work"):
        st.write(HOW_EVALUATORS)
    with st.expander("What the prototype does"):
        for i, (t, d) in enumerate(HOW_PROTOTYPE, 1):
            st.markdown(f"**{i}. {t}.** {d}")

    st.subheader("Settings")
    vals = {}
    cols = st.columns(len(SETTINGS))
    for c, s in zip(cols, SETTINGS):
        with c:
            if s["key"] == "seed":
                vals["seed"] = int(st.number_input(s["label"], min_value=1, max_value=999, value=1, step=1))
            else:
                vals[s["key"]] = st.select_slider(s["label"], options=s["options"], value=s["default"])
            st.caption(s["help"])
    engine = st.radio("Who proposes and writes the edits",
                      ["Offline (a lasso and a fixed edit library; free, about 20 seconds)",
                       "AI model (set it up in the sidebar; about an hour and roughly $10 with Claude Sonnet)"])
    if st.button("Run the prototype", type="primary"):
        from evloop.evaluators import make_world
        from evloop.loop import LoopConfig, TextSpace, run_panel
        from evloop.testbed import sample_applications
        k = max(1, min(5, 150 // vals["battery"]))
        apps = sample_applications(156, seed=vals["seed"])
        world = make_world(apps, seed=vals["seed"], delta=vals["delta"], rho=vals["rho"])
        cfg = LoopConfig(seed=vals["seed"], battery_apps=vals["battery"], battery_features=k, budget=k * vals["battery"])
        if engine.startswith("Offline"):
            with st.spinner("Running the prototype on 20 simulated evaluators..."):
                res = run_panel(world, TextSpace(apps, seed=vals["seed"]), cfg)
            st.session_state["sim"] = simulated_summary(world, res, "offline", vals)
        else:
            if not model_ready(ms):
                return
            from evloop.llm import LLMBackend
            client, out = make_client(ms, f"simulated_d{vals['delta']}_r{vals['rho']}_b{vals['battery']}_s{vals['seed']}")
            cfg.max_hypotheses, cfg.max_rounds = 6, 1
            with st.status("Running the prototype with the AI model...", expanded=True) as status:
                lines = st.empty()
                log = []

                def say(msg):
                    log.append(msg)
                    lines.code("\n".join(log[-12:]))
                backend = LLMBackend(client, world, n_twin_pairs=20, twin="ridge", workers=ms["workers"],
                                     proposer_prompt="v2", n_examples=20, log=say)
                res = run_panel(world, backend, cfg)
                status.update(label=f"Done: {client.n_calls} model calls", state="complete")
            st.session_state["sim"] = simulated_summary(world, res, "ai", vals)
    if "sim" in st.session_state:
        show_simulated(st.session_state["sim"])


def page_upload(ms):
    from evloop.realdata import DataError, batteries_table, hypotheses_table, load_dataset
    st.header("Your own data")
    st.write("Upload applications and ratings from a panel where every evaluator rated the same applications. "
             "An AI model proposes, for each evaluator, features of the text that could explain how their ratings "
             "depart from the rest of the panel, rewrites applications to change only that feature, and a "
             "statistical copy of the evaluator screens the ideas. You get screened hypotheses per evaluator and the "
             "edited applications a re-rating study would use. Nothing is confirmed until evaluators re-rate.")
    with st.expander("File format"):
        st.table(pd.DataFrame(UPLOAD_FORMAT, columns=["File", "Columns"]))
        st.caption("Use the column names exactly as shown. Scores can be on any numeric scale. If gender columns are "
                   "given, each evaluator's same-gender premium is reported.")
        if st.button("Make example files from a simulated panel"):
            from evloop.evaluators import make_world
            from evloop.realdata import synthetic_to_frames
            from evloop.testbed import sample_applications
            A, R, V = synthetic_to_frames(make_world(sample_applications(156, seed=1), seed=1))
            st.session_state["examples"] = (A.to_csv(index=False), R.to_csv(index=False), V.to_csv(index=False))
        if "examples" in st.session_state:
            A, R, V = st.session_state["examples"]
            c = st.columns(3)
            c[0].download_button("applications.csv", A, "applications.csv", "text/csv")
            c[1].download_button("ratings.csv", R, "ratings.csv", "text/csv")
            c[2].download_button("evaluators.csv", V, "evaluators.csv", "text/csv")

    c = st.columns(3)
    fa = c[0].file_uploader("Applications", type="csv")
    fr = c[1].file_uploader("Ratings", type="csv")
    fv = c[2].file_uploader("Evaluators (optional)", type="csv")
    if not (fa and fr):
        return
    try:
        data = load_dataset(io.StringIO(fa.getvalue().decode("utf-8")), io.StringIO(fr.getvalue().decode("utf-8")),
                            io.StringIO(fv.getvalue().decode("utf-8")) if fv else None)
    except DataError as e:
        st.error(str(e))
        return
    st.success(f"Loaded {len(data.apps)} applications rated by {len(data.panel)} evaluators.")
    for n in data.notes:
        st.warning(n)

    with st.expander("Quick look without an AI model: words linked to each evaluator's departures (not tested)"):
        from evloop.loop import TextSpace, consensus_residual
        from evloop.realdata import quick_candidates
        j = st.selectbox("Evaluator", range(len(data.panel)), format_func=lambda i: data.panel[i].label, key="ql")
        resid, _ = consensus_residual(data.scores, j)
        cand = quick_candidates(TextSpace(data.apps), resid)
        cc = st.columns(2)
        cc[0].markdown("**More common where this evaluator scores higher than the panel**")
        cc[0].dataframe(pd.DataFrame(cand["higher"], columns=["Word or phrase", "Correlation", "p"]), hide_index=True)
        cc[1].markdown("**More common where this evaluator scores lower**")
        cc[1].dataframe(pd.DataFrame(cand["lower"], columns=["Word or phrase", "Correlation", "p"]), hide_index=True)
        st.caption("These are correlations in the existing ratings. Words that travel together cannot be told apart "
                   "here, which is what the edits and re-ratings are for.")

    st.subheader("Run with an AI model")
    c = st.columns(3)
    n_hyp = c[0].slider("Hypotheses per evaluator", 2, 8, 6)
    n_pairs = c[1].slider("Edit pairs per hypothesis for screening", 6, 30, 20)
    n_rerate = c[2].select_slider("Edited applications per re-rating battery", [16, 20, 30, 40], 30)
    est = len(data.panel) * n_hyp * n_pairs * 2 + 5 * n_rerate * 4
    st.caption(f"Roughly {est:,} model calls. With Claude Sonnet, the simulated runs cost about $10 per 5,000 calls; "
               "longer applications cost more.")
    if st.button("Propose and screen hypotheses", type="primary"):
        if not model_ready(ms):
            return
        from evloop.llm import LLMBackend
        from evloop.loop import LoopConfig
        from evloop.realdata import screen_dataset
        client, out = make_client(ms, "upload")
        cfg = LoopConfig(battery_apps=n_rerate, battery_features=5, max_hypotheses=n_hyp)
        bar = st.progress(0.0, "Starting")
        with st.status("Working...", expanded=False) as status:
            lines = st.empty()
            log = []

            def say(msg):
                log.append(msg)
                lines.code("\n".join(log[-12:]))
            backend = LLMBackend(client, data, n_twin_pairs=n_pairs, twin="ridge", workers=ms["workers"],
                                 proposer_prompt="v2", n_examples=min(20, len(data.apps) // 4), log=say)
            sr = screen_dataset(data, backend, cfg, progress=lambda f, m: bar.progress(min(f, 1.0), m), log=say)
            status.update(label=f"Done: {client.n_calls} model calls", state="complete")
        H, B = hypotheses_table(data, sr), batteries_table(sr)
        H.to_csv(os.path.join(out, "hypotheses.csv"), index=False)
        B.to_csv(os.path.join(out, "edits_to_rerate.csv"), index=False)
        st.session_state["upload"] = (upload_summary(data, sr), H, B)
    if "upload" in st.session_state:
        show_upload(*st.session_state["upload"])


def main():
    st.sidebar.title("Evaluator loop")
    st.sidebar.caption("A prototype that proposes, builds and tests explanations of individual evaluators' judgments.")
    page = st.sidebar.radio("Mode", ["Simulated panel", "Your own data"])
    ms = model_settings()
    if page == "Simulated panel":
        page_simulated(ms)
    else:
        page_upload(ms)


main()

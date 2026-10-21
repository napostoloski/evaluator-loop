"""Figures and a markdown report from benchmark_rows.csv."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .benchmark import ability_table, effect_table, recall_by_role, recovery_table, root_table

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"loop": "#2a78d6", "edit_all": "#eb6834", "twin": "#1baf7a", "library": "#eda100"}
LABELS = {"loop": "Full loop (propose, screen, then re-rate)", "edit_all": "Edit everything (re-rate all ten features)",
          "twin": "Twin only (no re-ratings)", "library": "Regression on all coded features (no re-ratings)"}
SHORT = {"loop": "Full loop", "loop_total": "Loop, total-reaction test", "edit_all": "Edit everything",
         "loop_het": "Loop, test allows for evaluator spread", "loop_rtm": "Loop, regression-to-mean covariate",
         "twin": "Twin only",
         "screen": "Screen only", "coded": "Coded covariates", "library": "Library regression",
         "redaction": "Redaction battery"}
ORDER = ["loop", "loop_total", "edit_all", "twin", "screen", "library", "coded"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _line(ax, x, y, key, label_end=True):
    ax.plot(x, y, color=COLORS[key], lw=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", ms=8, mec=SURFACE, mew=2, label=LABELS[key], zorder=3)
    if label_end:
        ax.annotate(SHORT[key], (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT2)


def fig_by_delta(rec: pd.DataFrame, path: str, budget: int = 150):
    d = rec[(rec.budget == budget) & (rec.battery_apps == 30) & (rec.rho == 0.6)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), facecolor=SURFACE, sharey=True)
    for ax, metric, title in zip(axes, ["recall", "precision"],
                                 ["Recall: planted rules found", "Precision: accepted rules that are real"]):
        _style(ax)
        for key in ["loop", "edit_all", "twin", "library"]:
            s = d[d.method == key].sort_values("delta")
            # end labels only where the lines are well separated; the legend carries identity
            _line(ax, list(s.delta), list(s[metric]), key, label_end=(metric == "precision"))
        ax.set_title(title, fontsize=10.5, color=TEXT, loc="left", pad=10)
        ax.set_xlabel("Planted effect size (points on the 1-5 scale)", fontsize=9, color=TEXT2)
        ax.set_xticks(sorted(d.delta.unique()))
        ax.set_xlim(min(d.delta) - 0.05, max(d.delta) + 0.22)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Share", fontsize=9, color=TEXT2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=TEXT2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _per_feature(r):
    return r.battery_apps if r.method == "loop" else r.budget // 10


def fig_budget(rec: pd.DataFrame, path: str, delta: float = 0.5):
    d = rec[rec.method.isin(["loop", "edit_all"]) & (rec.delta == delta) & (rec.rho == 0.6)].copy()
    d["per_feature"] = d.apply(_per_feature, axis=1)
    d = d.drop_duplicates(["method", "per_feature", "budget"])
    names = {"loop": "Full loop: screened features, 3 to 5 tested", "edit_all": "Edit everything: all ten features"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), facecolor=SURFACE, sharey=True)
    for ax, metric, title in zip(axes, ["recall", "precision"], ["Recall", "Precision"]):
        _style(ax)
        for key in ["loop", "edit_all"]:
            s_ = d[d.method == key].sort_values("per_feature")
            ax.plot(s_.per_feature, s_[metric], color=COLORS[key], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                    label=names[key])
        ax.set_title(f"{title} by re-ratings per tested feature", fontsize=10.5, color=TEXT, loc="left", pad=10)
        ax.set_xlabel("Applications each evaluator re-rates per tested feature", fontsize=9, color=TEXT2)
        ax.set_xticks(sorted(d.per_feature.unique()))
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Share", fontsize=9, color=TEXT2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=TEXT2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt=".2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _fmt_se(est, se):
    if pd.isna(est):
        return "n/a (no rules accepted)"
    return f"{est:.2f} ({se:.2f})"


def build_report(rows_csv: str, out_dir: str, budget: int = 150, delta: float = 0.5) -> str:
    # "null" is a role name here, not a missing value
    df = pd.read_csv(rows_csv, keep_default_na=False, na_values=["", "nan", "NaN"])
    rec = recovery_table(df)
    rec.to_csv(os.path.join(out_dir, "summary_recovery.csv"), index=False)
    roles = recall_by_role(df)
    roles.to_csv(os.path.join(out_dir, "summary_recall_by_role.csv"), index=False)
    roots = root_table(df, budget)
    roots.to_csv(os.path.join(out_dir, "summary_root_attribution.csv"), index=False)
    eff = effect_table(df, budget)
    eff.to_csv(os.path.join(out_dir, "summary_effect_estimates.csv"), index=False)
    abil = ability_table(df, budget)
    abil.to_csv(os.path.join(out_dir, "summary_ability_vs_bias.csv"), index=False)

    fig_by_delta(rec, os.path.join(out_dir, "fig_recovery_by_effect_size.png"), budget)
    fig_budget(rec, os.path.join(out_dir, "fig_budget.png"), delta)

    n_rep = int(df.seed.nunique())
    base = rec[(rec.rho == 0.6)]
    main = base[(base.budget == budget) & (base.battery_apps == 30) & base.method.isin(ORDER)].copy()
    main["order"] = main.method.map({m: i for i, m in enumerate(ORDER)})
    main = main.sort_values(["delta", "order"])
    main["Precision (SE)"] = [_fmt_se(a, b) for a, b in zip(main.precision, main.precision_se)]
    main["Recall (SE)"] = [_fmt_se(a, b) for a, b in zip(main.recall, main.recall_se)]
    main["Method"] = main.method.map(SHORT)
    main = main[["Method", "delta", "Precision (SE)", "Recall (SE)", "exact_share", "null_false_positive_rate",
                 "queries_per_evaluator"]]
    main.columns = ["Method", "Effect size", "Precision (SE)", "Recall (SE)", "Evaluators exactly right",
                    "Null evaluators given a rule", "Re-ratings per evaluator"]

    bud = base[base.method.isin(["loop", "edit_all"]) & (base.delta == delta)].copy()
    bud["per_feature"] = bud.apply(_per_feature, axis=1)
    bud = bud.drop_duplicates(["method", "per_feature", "budget"])
    bud = bud.sort_values(["method", "per_feature"], ascending=[False, True])
    bud["method"] = bud.method.map(SHORT)
    bud = bud[["method", "per_feature", "budget", "precision", "recall", "exact_share", "null_false_positive_rate"]]
    bud.columns = ["Method", "Re-ratings per tested feature", "Budget per evaluator", "Precision", "Recall",
                   "Evaluators exactly right", "Null evaluators given a rule"]

    rho = rec[rec.method.isin(["loop", "edit_all", "library"]) & (rec.delta == delta) & (rec.budget == budget)
              & (rec.battery_apps == 30)].sort_values(["method", "rho"])
    rho["method"] = rho.method.map(SHORT)
    rho = rho[["method", "rho", "precision", "recall", "null_false_positive_rate"]]
    rho.columns = ["Method", "Test-retest correlation of evaluators' reading (rho)", "Precision", "Recall",
                   "Null evaluators given a rule"]

    var = rec[rec.method.isin(["loop", "loop_het", "loop_rtm"]) & (rec.delta == delta) & (rec.budget == budget)
              & (rec.battery_apps == 30) & (rec.rho == 0.6)].copy()
    var["method"] = var.method.map(SHORT)
    var = var[["method", "precision", "recall", "exact_share", "null_false_positive_rate"]]
    var.columns = ["Confirmation test", "Precision", "Recall", "Evaluators exactly right", "Null evaluators given a rule"]

    rr = roles[(roles.budget == budget) & (roles.battery_apps == 30) & (roles.rho == 0.6) & (roles.delta == delta)].pivot(
        index="role", columns="method", values="recall")
    rr = rr[[c for c in ORDER if c in rr.columns]].rename(columns=SHORT).reset_index()
    rt = roots.copy()
    rt["order"] = rt.method.map({**{m: i for i, m in enumerate(ORDER)}, "redaction": 9})
    rt = rt.sort_values(["delta", "order"]).drop(columns="order")
    rt["method"] = rt.method.map(SHORT)
    rt.columns = ["Method", "Effect size", "Evaluators with apparent premium", "Root (cue or content) correct",
                  "Names the right feature"]
    ef = eff.copy()
    ef["method"] = ef.method.map(SHORT)
    ef.columns = ["Method", "Planted effect", "Mean estimate", "SD of estimate", "Rules"]
    ab = abil.copy()
    ab.columns = ["Effect size", "True type", "Share labelled ability", "Rules"]

    lines = [
        "# Planted-rule benchmark: results",
        "",
        f"{n_rep} replications per cell (benchmark seeds 100 to {99 + n_rep}, disjoint from the seeds used "
        "in development and tests). Each replication draws 156 synthetic applications and a panel of 20 "
        "synthetic evaluators (10 women, 10 men). Every evaluator shares four reactions (paying customers "
        "+0.30, fluent English +0.15, five or more years of experience +0.15, solo founder -0.15, each "
        "varying by evaluator with SD 0.10); 14 of the 20 also carry planted rules of the stated effect size. "
        "Re-ratings carry fresh noise: an evaluator's reading of a venture on the second occasion correlates "
        "rho = 0.6 with the first (test-retest correlation of ratings about 0.7). A rule counts as found only "
        "if the method reports the planted feature in the planted direction; shared reactions reported as an "
        "evaluator's own rule count as false. Main design: the loop re-rates up to five screened features, "
        f"30 applications each, so {budget} re-ratings per evaluator; edit everything spreads the same "
        "budget over all ten features (15 applications each); no evaluator re-rates an application twice.",
        "",
        "![Recovery by effect size](fig_recovery_by_effect_size.png)",
        "",
        "## Rule recovery",
        "",
        "Pooled over evaluators and replications; standard errors across replications in parentheses.",
        "",
        _md(main),
        "",
        "## Re-ratings per tested feature (effect size %.2f)" % delta,
        "",
        "With 156 applications and each re-rated at most once per evaluator, testing all ten features leaves "
        "at most 15 re-ratings per feature. Screening first lets the loop spend 30 to 50 on each feature it "
        "tests. Under realistic re-rating noise, the number of re-ratings per tested feature is what decides "
        "whether an individual evaluator's rule can be confirmed.",
        "",
        "![Budget](fig_budget.png)",
        "",
        _md(bud),
        "",
        "## Sensitivity to re-rating noise (effect size %.2f)" % delta,
        "",
        _md(rho),
        "",
        "## Variants of the confirmation test (effect size %.2f)" % delta,
        "",
        "The default test compares each evaluator's reaction with the panel median. One variant adds the "
        "estimated spread of reactions across evaluators to the standard error, so a rule must stand out "
        "from ordinary differences between evaluators. The other adds the evaluator's original residual as a "
        "covariate to remove regression to the mean from hypothesis selection; because that residual also "
        "contains the rule itself, it attenuates true effects.",
        "",
        _md(var),
        "",
        "## Recall by planted role (effect size %.2f)" % delta,
        "",
        _md(rr),
        "",
        "## Locating the root of an apparent same-gender premium",
        "",
        "Evaluators whose residual scores show a significant same-gender premium (p < 0.05) and whose premium "
        "has a planted root: a name-and-pronoun cue, or gender-correlated content (sector or home-based). "
        "\"Root correct\" means cue versus content is classified correctly. \"Names the right feature\" also "
        "requires the content feature itself to be identified, with no cue rule. The redaction battery can only "
        "say whether the premium runs through the cue.",
        "",
        _md(rt),
        "",
        "## Effect estimates for correctly recovered rules",
        "",
        "Estimates are conditional on the rule having been found, so small effects are overstated (only large "
        "draws pass), and large ones are slightly compressed by the 1 to 5 scale.",
        "",
        _md(ef),
        "",
        "## Ability or bias (outcome test on correctly recovered rules)",
        "",
        "A recovered rule is labelled ability if the rewarded feature predicts the outcome beyond the consensus "
        "score. In this world only paying customers carries such information, so the correct share is 1 for "
        "ability rules and 0 for bias rules. With 156 startups and noisy outcomes the test has little power.",
        "",
        _md(ab),
        "",
    ]
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path

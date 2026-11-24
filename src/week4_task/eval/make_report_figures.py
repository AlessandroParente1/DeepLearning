"""Make every figure and table for the report from the saved result files.

    uv run python eval/make_report_figures.py        (from the repo root)

Writes PNGs and CSV tables to report/figures/. Nothing is trained here, it only reads results/.
Colours are the Okabe-Ito colour-blind-safe set.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.environ.get("REPORT_FIG_DIR", "report/figures")  # where the PNGs and tables go
BLUE, ORANGE, GREEN, RED, GREY, BLACK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#888888", "#222222"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.titlesize": 8, "figure.dpi": 150, "savefig.dpi": 300})

# Losses measured on the saved checkpoints (cross-entropy per token, nats). Val/test for the three models that were
# trained before the loss histories were being saved were measured afterwards from the checkpoints.
LOSSES = [  # name, val, test, parameters
    ("Uniform guess (21 tokens)", 3.0445, 3.0445, 0),
    ("MLP, context 3", 2.3597, 2.3534, 11_927),
    ("MLP, context 32", 1.6639, 1.6521, 113_815),
    ("Transformer small_a", 1.6607, 1.6857, 419_223),
    ("Transformer small_b", 1.5024, 1.5095, 3_204_375),
    ("Transformer big", 1.5132, 1.5089, 19_000_000),
]


def _add_conditional_row():
    """The conditional (classifier-free) model: loss WITH the true family label, at the epoch chosen by early stopping
    (the mean of the loss with and without the label), from results/lm/small_b_cfg_history.csv."""
    path = "results/lm/small_b_cfg_history.csv"
    if not os.path.exists(path):
        return
    h = pd.read_csv(path)
    test = h[h["epoch"] == "test"].iloc[0]
    h = h[h["epoch"] != "test"].astype({"val_loss": float, "val_loss_null_label": float})
    best = h.loc[((h["val_loss"] + h["val_loss_null_label"]) / 2).idxmin()]
    LOSSES.append(("Transformer small_b + label", float(best["val_loss"]), float(test["val_loss"]), 3_205_399))


_add_conditional_row()


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


def fig_loss_comparison():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    names = [r[0] for r in LOSSES][::-1]
    tests = [r[2] for r in LOSSES][::-1]
    colours = [GREY if "Uniform" in n else ORANGE if "MLP" in n else BLUE for n in names]
    bars = ax.barh(names, tests, color=colours, height=0.6)
    for bar, value in zip(bars, tests):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", color=BLACK)
    ax.set_xlabel("test loss (lower is better)")
    ax.set_xlim(0, 3.5)
    save(fig, "fig1_loss_comparison")
    pd.DataFrame(LOSSES, columns=["model", "val_loss", "test_loss", "parameters"]).to_csv(f"{OUT}/table_losses.csv", index=False)


def fig_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=True)
    for ax, name, title in zip(axes, ["small_a", "small_b"], ["small_a (0.4M params)", "small_b (3.2M params)"]):
        h = pd.read_csv(f"results/lm/{name}_history.csv")
        h = h[h["epoch"] != "test"].astype({"epoch": int, "train_loss": float, "val_loss": float})
        ax.plot(h["epoch"], h["train_loss"], color=BLUE, label="train")
        ax.plot(h["epoch"], h["val_loss"], color=ORANGE, label="validation")
        best = h.loc[h["val_loss"].idxmin()]
        ax.scatter([best["epoch"]], [best["val_loss"]], color=ORANGE, zorder=3, s=14)
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False)
    save(fig, "fig2_training_curves")


def fig_novelty_vs_plddt():
    sets = [("MLP ctx 3", "mlp_data_01_sample", ORANGE), ("MLP ctx 32 (inferred)", "mlp_data_02_sample", ORANGE),
            ("Transformer T=1", "transformer_data_01_sample", BLUE),
            ("Transformer T=2", "transformer_data_02_temperature_sample", BLUE),
            ("Transformer T=0.5", "transformer_data_03_templow_samples", BLUE)]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.axvline(0.70, color=GREEN, lw=1, ls="--")
    ax.axvline(0.25, color=GREY, lw=1, ls="--")
    ax.text(0.71, 36, "real proteins\n(0.70)", color=GREEN, fontsize=7, va="bottom")
    ax.text(0.26, 36, "shuffled\n(0.25)", color=GREY, fontsize=7, va="bottom")
    rows = []
    for label, folder, colour in sets:
        s = pd.read_csv(f"results/eval/{folder}/summary.csv").iloc[0]
        ax.scatter(s["nearest_identity_median"], s["mean_plddt"], color=colour, s=28, edgecolor="white", zorder=3)
        offset = {"MLP ctx 3": (-6, -3), "MLP ctx 32 (inferred)": (6, 3), "Transformer T=1": (-6, -3),
                  "Transformer T=2": (6, 5), "Transformer T=0.5": (-6, 4)}[label]
        ax.annotate(label, (s["nearest_identity_median"], s["mean_plddt"]), textcoords="offset points",
                    xytext=offset, fontsize=7, ha="left" if offset[0] > 0 else "right", color=BLACK)
        rows.append({"set": label, **s.to_dict()})
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(35, 100)
    ax.set_xlabel("nearest identity to training set (median)")
    ax.set_ylabel("ESMFold pLDDT (mean)")
    save(fig, "fig3_novelty_vs_plddt")
    pd.DataFrame(rows).to_csv(f"{OUT}/table_samples.csv", index=False)


def fig_predictor_by_prefix():
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for folder, label, colour in [("classifier", "head on big transformer", BLUE),
                                  ("classifier_small_b", "head on small_b", ORANGE)]:
        d = pd.read_csv(f"results/{folder}/accuracy_by_prefix_test.csv")
        ax.plot(d["prefix_length"], d["accuracy"], color=colour, label=label, lw=1.4)
    ax.axhline(1 / 3, color=GREY, ls="--", lw=1)
    ax.text(100, 0.35, "chance", color=GREY, fontsize=7, ha="right")
    ax.set_xlabel("residues of the protein seen so far")
    ax.set_ylabel("family accuracy (test)")
    ax.set_ylim(0.25, 1.02)
    ax.legend(frameon=False, loc="center right")
    save(fig, "fig4_predictor_by_prefix")


def fig_guidance():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    rows = []
    for tag, label, colour in [("guided_HU", "big transformer", BLUE), ("guided_small_b_HU", "small_b", ORANGE)]:
        d = pd.read_csv(f"results/sequences/{tag}_family_check.csv")
        x = range(len(d))
        ax.plot(x, d["head_says_target"], color=colour, marker="o", ms=4, lw=1.4, label=f"{label}: guiding head")
        ax.plot(x, d["neighbour_is_target"], color=colour, marker="s", ms=4, lw=1.2, ls="--",
                label=f"{label}: nearest training protein")
        ax.set_xticks(list(x), [f"{g:g}" for g in d["gamma"]])
        rows.append(d.assign(backbone=label))
    ax.axhline(1 / 3, color=GREY, ls=":", lw=1)
    ax.set_xlabel(r"guidance strength $\gamma$")
    ax.set_ylabel("share of samples that are HU")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig5_guidance_vs_gamma")
    pd.concat(rows).to_csv(f"{OUT}/table_guidance.csv", index=False)


def fig_cg_vs_cfg():
    """Classifier guidance (gamma) against classifier-free guidance (w), same axes: three panels, full page width."""
    cg_path = "results/sequences/guided_small_b_HU_n256_family_check.csv"
    cg = pd.read_csv(cg_path if os.path.exists(cg_path) else "results/sequences/guided_small_b_HU_family_check.csv")
    cg = cg.rename(columns={"gamma": "strength"})
    cfg = pd.read_csv("results/sequences/cfg_small_b_HU_n256_family_check.csv")
    real = pd.read_csv("results/sequences/control_real.csv")["sequence"].str.len().mean()
    series = [(cg, r"classifier guidance ($\gamma$)", BLUE, "o"), (cfg, r"classifier-free guidance ($w$)", ORANGE, "s")]
    panels = [("neighbour_is_target", "share of samples that are HU"),
              ("median_nearest_identity", "nearest identity to training set"),
              ("mean_length", "mean length (residues)")]
    ticks = [0, 0.5, 1, 2, 4, 8, 16]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9}):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
        for ax, (column, label) in zip(axes, panels):
            for frame, name, colour, marker in series:
                ax.plot(frame["strength"], frame[column], color=colour, marker=marker, ms=4, lw=1.4, label=name)
            ax.set_xscale("symlog", linthresh=0.5)
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("guidance strength")
            ax.set_ylabel(label)
        axes[0].axhline(1 / 3, color=GREY, ls=":", lw=1)
        axes[0].set_ylim(0, 1.05)
        axes[1].axhline(0.70, color=GREEN, ls="--", lw=1)
        axes[1].axhline(0.25, color=GREY, ls="--", lw=1)
        axes[1].text(0.2, 0.68, "real proteins", color=GREEN, fontsize=8, va="top")
        axes[1].text(0.2, 0.27, "shuffled", color=GREY, fontsize=8, va="bottom")
        axes[1].set_ylim(0, 1)
        axes[2].axhline(real, color=GREEN, ls="--", lw=1)
        axes[2].text(0.2, real + 2, "real proteins", color=GREEN, fontsize=8, va="bottom")
        axes[2].set_ylim(0, 118)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(f"{OUT}/fig6_cg_vs_cfg.png")
        plt.close(fig)
        print("wrote", f"{OUT}/fig6_cg_vs_cfg.png")
    pd.concat([cg.assign(method="classifier guidance"), cfg.assign(method="classifier-free guidance")]).to_csv(
        f"{OUT}/table_cg_vs_cfg.csv", index=False)


def _folded(eval_name, family_check_name):
    """Per-sample ESMFold pLDDT (results/eval/<eval_name>/eval.csv) joined with the nearest-training-protein family check,
    one row per sample. Both files keep the sample order, which is asserted."""
    e = pd.read_csv(f"results/eval/{eval_name}/eval.csv")
    f = pd.read_csv(f"results/sequences/{family_check_name}_family_check_per_sample.csv")
    f = f.groupby("setting", sort=False).head(len(e) // e["setting"].nunique()).reset_index(drop=True)
    assert len(e) == len(f) and (e["sequence"].values == f["sequence"].values).all(), "sample order differs"
    e["neighbour_is_target"] = f["neighbour_is_target"].values
    e["strength"] = e["gamma"] if "gamma" in e else e["w"]
    return e


def _fold_summary(e):
    g = e.groupby("strength", sort=True)
    n = g.size()
    return pd.DataFrame({"n": n, "mean_plddt": g["mean_plddt"].mean(), "se": g["mean_plddt"].std() / n ** 0.5,
                         "share_below_70": g["mean_plddt"].apply(lambda s: (s < 70).mean()),
                         "share_HU": g["neighbour_is_target"].mean(), "median_identity": g["nearest_identity"].median()})


def fig_plddt_and_tradeoff():
    """Structure plausibility (ESMFold pLDDT) against guidance strength, and against the share of samples that are HU."""
    cg = _fold_summary(_folded("guided_small_b_HU_n256_f256", "guided_small_b_HU_n256"))
    cfg = _fold_summary(_folded("cfg_small_b_HU_n256_f256", "cfg_small_b_HU_n256"))
    real = pd.read_csv("results/eval/control_real_f256/eval.csv")["mean_plddt"]
    shuffled = pd.read_csv("results/eval/control_shuffled_f256/eval.csv")["mean_plddt"]
    series = [(cg, r"classifier guidance ($\gamma$)", BLUE, "o"), (cfg, r"classifier-free guidance ($w$)", ORANGE, "s")]

    # fig7: pLDDT against strength, mean +- 1 standard error, with the two controls as bands (mean +- 1 SE)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for frame, name, colour, marker in series:
        ax.errorbar(frame.index, frame["mean_plddt"], yerr=frame["se"], color=colour, marker=marker, ms=4, lw=1.4,
                    capsize=2, label=name)
    for values, name, colour, y in [(real, "real proteins", GREEN, 93), (shuffled, "shuffled", GREY, 43)]:
        m, se = values.mean(), values.std() / len(values) ** 0.5
        ax.axhline(m, color=colour, ls="--", lw=1)
        ax.axhspan(m - se, m + se, color=colour, alpha=0.2, lw=0)
        ax.text(0.2, y, name, color=colour, fontsize=7, va="center")
    ax.set_xscale("symlog", linthresh=0.5)
    ticks = [0, 0.5, 1, 2, 4, 8, 16]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(30, 100)
    ax.set_xlabel("guidance strength")
    ax.set_ylabel("ESMFold pLDDT (mean, 256 per point)")
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.45))
    save(fig, "fig7_plddt_vs_strength")

    # fig8: the family-versus-structure picture. x: how many samples are HU, y: how plausible the structures are
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for frame, name, colour, marker in series:
        ax.errorbar(frame["share_HU"], frame["mean_plddt"], yerr=frame["se"], color=colour, marker=marker, ms=4,
                    lw=0.8, capsize=2, label=name)
    def note(text, xy, xytext, ha="center"):
        ax.annotate(text, xy, xytext=xytext, fontsize=7, color=BLACK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6))

    note(r"$\gamma=0$", (cg.loc[0, "share_HU"], cg.loc[0, "mean_plddt"]), (0.36, 82))
    note(r"$\gamma=1,\,8$", (0.975, 86.9), (0.74, 83))
    note(r"$\gamma\geq 2$, $w\leq 4$", (0.998, 91.2), (0.80, 96))
    note(r"$w=8$", (cfg.loc[8, "share_HU"], cfg.loc[8, "mean_plddt"]), (0.90, 80.5))
    note(r"$w=16$", (cfg.loc[16, "share_HU"], cfg.loc[16, "mean_plddt"]), (0.90, 67))
    ax.axhline(real.mean(), color=GREEN, ls="--", lw=1)
    ax.text(0.31, real.mean() + 1.5, "real proteins", color=GREEN, fontsize=7)
    ax.set_xlabel("share of samples that are HU")
    ax.set_ylabel("ESMFold pLDDT (mean)")
    ax.set_ylim(60, 100)
    ax.set_xlim(0.28, 1.04)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    save(fig, "fig8_family_vs_structure")

    table = pd.concat([cg.assign(method="classifier guidance"), cfg.assign(method="classifier-free guidance")])
    table.index.name = "strength"
    table.to_csv(f"{OUT}/table_plddt.csv")
    pd.DataFrame({"set": ["real proteins", "shuffled"], "mean_plddt": [real.mean(), shuffled.mean()],
                  "se": [real.std() / len(real) ** 0.5, shuffled.std() / len(shuffled) ** 0.5],
                  "n": [len(real), len(shuffled)]}).to_csv(f"{OUT}/table_plddt_controls.csv", index=False)


if __name__ == "__main__":
    fig_plddt_and_tradeoff()
    fig_loss_comparison()
    fig_training_curves()
    fig_novelty_vs_plddt()
    fig_predictor_by_prefix()
    fig_guidance()
    fig_cg_vs_cfg()

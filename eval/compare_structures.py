"""Compare the ESMFold structures of several sample sets: fold, secondary structure, compactness.

Run `eval/eval.py --fold` for every set first. A set is one of its output directories, holding
pdb/ and eval.csv, and is named on the command line. The measurements are
`protgen.metrics.structures`; this script is the comparison across sets and the figure.

Per structure:

  fold          best TM-score to a panel of reference structures, after a sequence-independent
                structural alignment (the 3Di idea Foldseek uses). The panel is a random sample of
                confident structures (pLDDT >= 70) from the --reference set, so list the real
                control first. TM > 0.5 usually means the same fold.
  secondary     fraction helix / strand / coil (P-SEA) and the order of the helices (a) and
  structure     strands (b), e.g. "babbab"
  compactness   radius of gyration / (2.2 N^0.38): about 1 for a compact globular protein
  confidence    mean pLDDT of the first 15, the middle and the last 15 residues

Per set it also estimates structural diversity as the mean pairwise TM-score in a random sample
(lower = more diverse). A structure shorter than --min-length is a failure rather than a
measurement - a handful of atoms always superimposes well - so it is left out of the TM-scores and
the figure, and counted in frac_shorter_than_min_length instead.

Writes structures.csv (one row per structure), summary.csv (one row per set) and comparison.png.

Example:
    uv run --group eval python eval/compare_structures.py --out results/structure_comparison \\
        --sets real=results/eval/control_real shuffled=results/eval/control_shuffled \\
               gru=results/eval/gen_gru
"""

import argparse
import itertools
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protgen.metrics.structures import POSITION_BINS, best_tm, describe, init_panel, pair_tm
from protgen.metrics.structures import summarize as summarize_structures

# Figure tokens. Series colours are assigned in this fixed order, one per set as given.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def plot(df, profiles, colors, out):
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
    df = df[df.n_residues >= df.attrs["min_length"]]
    bins = np.arange(0, 1.0001, 0.05)

    ax = axes[0]
    for name, color in colors.items():
        g = df[(df.set == name) & df.nearest_identity.notna()]
        if g.empty:
            continue
        ax.scatter(g.nearest_identity, g.mean_plddt, s=9, color=color, alpha=0.25, linewidths=0)
        idx = np.digitize(g.nearest_identity, bins)
        medians = [(g[idx == i].nearest_identity.median(), g[idx == i].mean_plddt.median())
                   for i in np.unique(idx) if np.sum(idx == i) >= 10]
        if medians:
            ax.plot(*zip(*medians), color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set(xlabel="Identity to nearest training sequence", ylabel="Mean pLDDT", xlim=(0, 1))
    ax.set_ylim(top=100)
    ax.set_title("A  pLDDT vs similarity to training data")

    ax = axes[1]
    x = (np.arange(POSITION_BINS) + 0.5) / POSITION_BINS
    for name, color in colors.items():
        p = profiles[name]
        if len(p) == 0:
            continue
        q25, q50, q75 = np.nanpercentile(p, [25, 50, 75], axis=0)
        ax.fill_between(x, q25, q75, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, q50, color=color, lw=2, label=name)
    ax.set(xlabel="Relative position in chain (N → C)", ylabel="Per-residue pLDDT (median, IQR band)", xlim=(0, 1))
    ax.set_ylim(top=100)
    ax.set_title("B  Confidence along the chain")

    ax = axes[2]
    for name, color in colors.items():
        tm = np.sort(df[(df.set == name) & ~df.in_panel].best_tm.dropna().values)
        if len(tm):
            ax.step(tm, np.arange(1, len(tm) + 1) / len(tm), where="post", color=color, lw=2)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(0.51, 0.97, "TM = 0.5\nsame fold", color=MUTED, fontsize=9, va="top")
    ax.set(xlabel=f"Best TM-score to {df.attrs['panel_size']} reference structures",
           ylabel="Cumulative fraction of structures", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("C  Fold similarity to the reference set")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False, labelcolor=INK2)
    fig.savefig(out, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", required=True, metavar="NAME=DIR",
                        help="eval.py output directories (with pdb/) to compare, "
                             "e.g. real=results/eval/control_real")
    parser.add_argument("--reference", help="set whose structures form the TM-score panel (default: the first set)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--panel-size", type=int, default=25)
    parser.add_argument("--panel-min-plddt", type=float, default=70, help="only confident structures enter the panel")
    parser.add_argument("--diversity-sample", type=int, default=50, help="structures per set for the pairwise TM-score")
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    sets = dict(s.split("=", 1) for s in args.sets)
    if len(sets) > len(COLORS):
        parser.error(f"at most {len(COLORS)} sets can be compared in one figure")
    reference = args.reference or next(iter(sets))
    if reference not in sets:
        parser.error(f"--reference {reference!r} is not one of the sets")
    paths = {name: sorted(Path(d, "pdb").glob("*.pdb")) for name, d in sets.items()}
    for name, p in paths.items():
        if not p:
            parser.error(f"no PDB files in {sets[name]}/pdb - run eval.py --fold on that set first")

    with Pool(args.workers) as pool:
        described = {name: pool.map(describe, p, chunksize=16) for name, p in paths.items()}
    frames = []
    for name, rows in described.items():
        g = pd.DataFrame([row for row, _ in rows])
        g.insert(0, "set", name)
        # The sequence metrics of the same samples, joined on the sample id: the PDB files are
        # named after the sample, so a structure and its sequence row stay together.
        metrics = Path(sets[name], "eval.csv")
        if metrics.exists():
            g = g.merge(pd.read_csv(metrics)[["id", "nearest_identity"]], on="id", how="left")
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    if "nearest_identity" not in df:
        df["nearest_identity"] = np.nan
    profiles = {name: np.array([prof for (row, prof) in rows if row["n_residues"] >= args.min_length])
                for name, rows in described.items()}

    rng = random.Random(args.seed)
    candidates = df[(df.set == reference) & (df.mean_plddt >= args.panel_min_plddt) & (df.n_residues >= args.min_length)]
    panel = rng.sample(list(candidates.path), min(args.panel_size, len(candidates)))
    df["in_panel"] = df.path.isin(panel)
    to_score = df[(df.n_residues >= args.min_length) & ~df.in_panel].path.tolist()
    print(f"Aligning {len(to_score)} structures against a panel of {len(panel)} '{reference}' structures ...", flush=True)
    with Pool(args.workers, initializer=init_panel, initargs=(panel,)) as pool:
        df["best_tm"] = df.path.map(dict(zip(to_score, pool.map(best_tm, to_score, chunksize=8))))

    pairwise = {}
    with Pool(args.workers) as pool:
        for name in sets:
            sample = df[(df.set == name) & (df.n_residues >= args.min_length)].path.tolist()
            sample = rng.sample(sample, min(args.diversity_sample, len(sample)))
            pairwise[name] = float(np.nanmean(pool.map(pair_tm, list(itertools.combinations(sample, 2)), chunksize=16)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "structures.csv", index=False)
    summary = pd.DataFrame({name: summarize_structures(df[df.set == name], args.min_length,
                                                      pairwise_tm=pairwise[name])
                            for name in sets})
    summary.T.to_csv(out / "summary.csv", index_label="set")
    (out / "settings.json").write_text(json.dumps({**vars(args), "panel": panel}, indent=2) + "\n")

    df.attrs.update(min_length=args.min_length, panel_size=len(panel))
    plot(df, profiles, dict(zip(sets, COLORS)), out / "comparison.png")

    print("\n" + summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote structures.csv, summary.csv, settings.json and comparison.png to {out}")


if __name__ == "__main__":
    main()

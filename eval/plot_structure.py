"""Visualise ESMFold structures coloured by pLDDT (stored in the PDB B-factor column).

For every PDB this writes
  <name>.png   Calpha trace coloured by pLDDT + per-residue pLDDT plot (static, works anywhere)
  <name>.html  interactive 3D cartoon (rotate with the mouse); needs py3Dmol, open it in a browser
               on your own computer after downloading it from the cluster

Colours follow the AlphaFold convention: blue > 90, light blue 70-90, yellow 50-70, orange < 50.

Example:
    uv run --group eval python eval/plot_structure.py results/eval/mock_data/pdb/sample_0.pdb
    uv run --group eval python eval/plot_structure.py results/eval/mock_data/pdb --limit 10         --out results/eval/mock_data/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from biotite.structure.io.pdb import PDBFile

BINS = [(90, "#0053D6", "very high (>90)"), (70, "#65CBF3", "confident (70-90)"),
        (50, "#FFDB13", "low (50-70)"), (-1, "#FF7D45", "very low (<50)")]


def color_for(plddt):
    return next(color for threshold, color, _ in BINS if plddt > threshold)


def load_ca(pdb_path):
    """Return Calpha coordinates (n, 3), pLDDT per residue (n,), residue numbers (n,)."""
    atoms = PDBFile.read(pdb_path).get_structure(model=1, extra_fields=["b_factor"])
    ca = atoms[atoms.atom_name == "CA"]
    return ca.coord, ca.b_factor, ca.res_id


def plot_png(pdb_path, out_path):
    coords, plddt, _ = load_ca(pdb_path)
    fig = plt.figure(figsize=(11, 4.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i in range(len(coords) - 1):
        segment = coords[i:i + 2]
        ax.plot(*segment.T, color=color_for(min(plddt[i], plddt[i + 1])), linewidth=3, solid_capstyle="round")
    centre, radius = coords.mean(axis=0), np.ptp(coords, axis=0).max() / 2
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        set_lim(c - radius, c + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(f"{Path(pdb_path).stem}  (mean pLDDT {plddt.mean():.1f})")

    ax = fig.add_subplot(1, 2, 2)
    positions = np.arange(1, len(plddt) + 1)
    for threshold, color, label in BINS:
        ax.axhspan(max(threshold, 0), {90: 100, 70: 90, 50: 70, -1: 50}[threshold], color=color, alpha=0.2, label=label)
    ax.plot(positions, plddt, color="black", linewidth=1.5)
    ax.set_xlim(1, len(plddt))
    ax.set_ylim(0, 100)
    ax.set_xlabel("residue")
    ax.set_ylabel("pLDDT")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_html(pdb_path, out_path):
    import py3Dmol

    _, plddt, res_ids = load_ca(pdb_path)
    view = py3Dmol.view(width=800, height=600)
    view.addModel(Path(pdb_path).read_text(), "pdb")
    for _, color, _ in BINS:
        residues = [int(r) for r, p in zip(res_ids, plddt) if color_for(p) == color]
        if residues:
            view.setStyle({"resi": residues}, {"cartoon": {"color": color}})
    view.zoomTo()
    html = view.write_html() if hasattr(view, "write_html") else view._make_html()
    Path(out_path).write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb", help="a PDB file, or a directory of PDB files")
    parser.add_argument("--out", help="output directory (default: next to the PDB files)")
    parser.add_argument("--limit", type=int, help="only plot the first N PDB files of a directory")
    parser.add_argument("--no-html", action="store_true", help="skip the interactive HTML (no py3Dmol needed)")
    args = parser.parse_args()

    path = Path(args.pdb)
    files = sorted(path.glob("*.pdb")) if path.is_dir() else [path]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No PDB files found in {path}")

    for pdb in files:
        out_dir = Path(args.out) if args.out else pdb.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_png(pdb, out_dir / f"{pdb.stem}.png")
        if not args.no_html:
            write_html(pdb, out_dir / f"{pdb.stem}.html")
        print(f"Wrote figures for {pdb.name} to {out_dir}")


if __name__ == "__main__":
    main()

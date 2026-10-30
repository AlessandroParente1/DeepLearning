import itertools
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

# Relative positions the per-residue pLDDT is averaged into, so that profiles of proteins of
# different lengths can be compared and plotted on one axis.
POSITION_BINS = 20
# A helix or strand shorter than this is not counted in the topology string: biotite's annotation
# produces single-residue elements that no one would call a helix.
MIN_SSE_LENGTH = {"a": 4, "b": 3}


def load(path):
    """The amino acid atoms of the first model of a PDB file, with the B-factor kept."""
    atoms = pdb.PDBFile.read(path).get_structure(model=1, extra_fields=["b_factor"])
    return atoms[struc.filter_amino_acids(atoms)]


def topology(sse):
    """Order of helices (a) and strands (b), e.g. "babbab", short elements ignored."""
    return "".join(key for key, group in itertools.groupby(sse)
                   if key in MIN_SSE_LENGTH and len(list(group)) >= MIN_SSE_LENGTH[key])


def describe(path):
    """(row, profile) for one structure: its metrics, and its pLDDT over POSITION_BINS positions.

    `row` keys on `id`, the PDB file's stem, which is the sample id the sequence table uses.

    compactness is the radius of gyration over 2.2 N^0.38, the expected radius of a globular
    protein of that length, so it is about 1 for a compact fold and well above it for an extended
    or disordered one.
    """
    atoms = load(path)
    ca = atoms[atoms.atom_name == "CA"]
    n = len(ca)
    sse = struc.annotate_sse(atoms)
    plddt = ca.b_factor
    position = np.minimum(np.arange(n) * POSITION_BINS // n, POSITION_BINS - 1)
    counts = np.bincount(position, minlength=POSITION_BINS)
    with np.errstate(invalid="ignore"):
        profile = np.bincount(position, weights=plddt, minlength=POSITION_BINS) / counts
    row = {
        "id": Path(path).stem, "n_residues": n, "mean_plddt": float(plddt.mean()),
        "frac_helix": float(np.mean(sse == "a")), "frac_strand": float(np.mean(sse == "b")),
        "frac_coil": float(np.mean(sse == "c")), "topology": topology(sse),
        "rg_norm": float(struc.gyration_radius(ca) / (2.2 * n ** 0.38)),
        "plddt_first15": float(plddt[:15].mean()), "plddt_last15": float(plddt[-15:].mean()),
        "plddt_core": float(plddt[15:-15].mean() if n > 30 else plddt.mean()),
        "path": str(path),
    }
    return row, profile


def tm_after_alignment(fixed, mobile, reference_length):
    """TM-score after one structural superposition, with no iterative refinement.

    Fast, and if anything it underestimates what TM-align would report, so a structure counted as
    the same fold here really is one.
    """
    fitted, _, fixed_idx, mobile_idx = struc.superimpose_structural_homologs(
        fixed, mobile, max_iterations=1)
    return struc.tm_score(fixed, fitted, fixed_idx, mobile_idx, reference_length=reference_length)


# The panel of reference structures every query is scored against. It is module state because the
# comparison runs in a multiprocessing Pool and the panel is loaded once per worker, through
# `init_panel` as the pool's initializer, rather than pickled with every task.
PANEL = []


def init_panel(paths):
    """Load the reference panel into this process. A Pool initializer."""
    global PANEL
    PANEL = [load(path) for path in paths]


def best_tm(path):
    """Best TM-score of one structure against the panel, normalised by its own length."""
    query = load(path)
    n = int(np.sum(query.atom_name == "CA"))
    scores = []
    for reference in PANEL:
        try:
            scores.append(tm_after_alignment(reference, query, n))
        except Exception:  # the alignment fails on degenerate structures; skip that reference
            pass
    return max(scores) if scores else float("nan")


def pair_tm(paths):
    """TM-score between two structures, normalised by the shorter one. For diversity."""
    try:
        return tm_after_alignment(load(paths[0]), load(paths[1]), "shorter")
    except Exception:
        return float("nan")


def summarize(structures, min_length, pairwise_tm=float("nan")):
    """The set-level view of a table of `describe` rows.

    `structures` needs the columns `describe` produces plus `in_panel`, because a panel structure
    would score a perfect TM-score against itself and is left out of the fold numbers. A structure
    shorter than `min_length` is a failure rather than a measurement - a handful of atoms always
    superimposes well - so it is excluded from everything but the failure fraction.
    """
    ok = structures[structures.n_residues >= min_length]
    scored = structures[~structures.in_panel]
    return {
        "n_structures": len(structures),
        "frac_shorter_than_min_length": float(np.mean(structures.n_residues < min_length)),
        "mean_plddt": structures.mean_plddt.mean(),
        "median_plddt": structures.mean_plddt.median(),
        "frac_plddt_gt70": float(np.mean(structures.mean_plddt > 70)),
        "frac_plddt_gt90": float(np.mean(structures.mean_plddt > 90)),
        "frac_same_fold_tm_gt0.5": float(np.mean(scored.best_tm > 0.5)),
        "median_best_tm": scored.best_tm.median(),
        "mean_pairwise_tm": pairwise_tm,
        "frac_helix": ok.frac_helix.mean(), "frac_strand": ok.frac_strand.mean(),
        "frac_coil": ok.frac_coil.mean(),
        "median_rg_norm": ok.rg_norm.median(),
        "frac_rg_norm_gt1.3": float(np.mean(ok.rg_norm > 1.3)),
        "median_plddt_first15": ok.plddt_first15.median(),
        "median_plddt_core": ok.plddt_core.median(),
        "median_plddt_last15": ok.plddt_last15.median(),
    }

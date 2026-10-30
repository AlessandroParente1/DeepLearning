import math
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal, get_sequence_identity

STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

MATRIX = SubstitutionMatrix.std_protein_matrix()
GAP_PENALTY = (-10, -1)
DEFAULT_TOP_K = 50

RADIX = len(STANDARD_RESIDUES)
INDEX = {residue: i for i, residue in enumerate(STANDARD_RESIDUES)}


def read_sequences(path):
    """One row per sequence from a CSV that has a `sequence` column. Nothing else is required.

    This is the one way every script here reads a table of sequences - the samples, the reference
    and the controls alike - so a sequence written in lower case is the same protein to all of
    them.

    `length` is computed here rather than trusted, because it is derived from the sequence and a
    column that disagrees with it would be read as the truth by every length-aware report. `id` is
    made up (`sample_0`, `sample_1`, ...) when absent, since the structures and the pLDDT join
    back on it. Any other column is carried through untouched.
    """
    path = Path(path)
    # skip_blank_lines=False because in a CSV whose only column is `sequence`, an empty sequence -
    # a model that emitted <EOS> straight away - is a blank line, and pandas drops those by
    # default. Dropping it would quietly shrink the denominator of every fraction below. A
    # trailing newline still does not add a row.
    frame = pd.read_csv(path, skip_blank_lines=False)
    if "sequence" not in frame.columns:
        raise ValueError(f"{path} has no `sequence` column; its columns are "
                         f"{', '.join(frame.columns)}")
    # An empty sequence is a blank field, which pandas reads as NaN. Left alone it would reach the
    # metrics as the float nan.
    frame["sequence"] = frame["sequence"].fillna("").astype(str).str.strip().str.upper()
    frame["length"] = frame["sequence"].str.len()
    if "id" in frame.columns:
        frame["id"] = frame["id"].astype(str)
    else:
        frame.insert(0, "id", [f"sample_{i}" for i in range(len(frame))])
    if "setting" not in frame.columns:
        frame["setting"] = path.stem
    return frame


def read_reference(path):
    """The reference sequences as a plain list of strings, from a CSV with a `sequence` column.

    Read through `read_sequences`, so the samples and the reference are normalised the same way,
    and then filtered: an empty or non-standard reference sequence is nothing a sample could be
    close to, and the alignment would be asked to align it anyway.
    """
    sequences = read_sequences(path)["sequence"].tolist()
    keep = [sequence for sequence in sequences if is_valid(sequence)]
    if len(keep) < len(sequences):
        print(f"  ignoring {len(sequences) - len(keep)} reference sequence(s) that are empty or "
              f"hold non-standard residues")
    if not keep:
        raise ValueError(f"No usable sequences in {path}")
    return keep


def write_sequences(path, rows):
    """Write sample rows - dicts or a DataFrame - as a CSV `read_sequences` can read back.

    Returns the path, and creates the directory if it is not there yet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if "sequence" not in frame.columns:
        raise ValueError("rows have no `sequence` column")
    frame.to_csv(path, index=False)
    return path


def is_valid(sequence):
    """Whether a sequence is non-empty and made only of the 20 standard residues."""
    return bool(sequence) and set(sequence) <= set(STANDARD_RESIDUES)


def identity(a, b):
    """Identical positions over the whole global alignment of two residue strings."""
    alignment = align_optimal(ProteinSequence(a), ProteinSequence(b), MATRIX,
                              gap_penalty=GAP_PENALTY, max_number=1)[0]
    return get_sequence_identity(alignment, mode="all")


def kmer_distribution(sequences, k):
    """{k-mer: frequency} over every k-mer of every sequence."""
    counts = Counter()
    for sequence in sequences:
        counts.update(sequence[i:i + k] for i in range(len(sequence) - k + 1))
    total = sum(counts.values())
    return {kmer: n / total for kmer, n in counts.items()}


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits between two {item: probability} dicts."""
    divergence = 0.0
    for item in set(p) | set(q):
        pi, qi = p.get(item, 0.0), q.get(item, 0.0)
        mi = (pi + qi) / 2
        if pi > 0:
            divergence += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            divergence += 0.5 * qi * math.log2(qi / mi)
    return divergence


def entropy_bits(sequence):
    """Shannon entropy of the residue composition of one sequence, in bits."""
    counts = Counter(sequence)
    return -sum(n / len(sequence) * math.log2(n / len(sequence)) for n in counts.values())


def longest_run(sequence):
    """Length of the longest run of one repeated residue, which flags `AAAAAA...` junk."""
    return max(len(list(group)) for _, group in groupby(sequence))


class NearestFinder:
    """The closest reference sequence to a query: 3-mer inverted index, then alignment.

    Aligning a query against every reference is too slow to do for a whole sample set, so the
    references are ranked by how many distinct 3-mers they share with the query and only the top
    `top_k` are aligned. The prefilter is strongest exactly where this matters: a sample that
    reproduces a training protein shares almost all of its 3-mers with it.
    """

    def __init__(self, reference, top_k=DEFAULT_TOP_K):
        self.reference, self.top_k = list(reference), top_k
        kmers, owners = [], []
        for position, sequence in enumerate(self.reference):
            ids = np.unique(self.encode(sequence))
            kmers.append(ids)
            owners.append(np.full(len(ids), position, dtype=np.int32))
        kmers, owners = np.concatenate(kmers), np.concatenate(owners)
        order = np.argsort(kmers, kind="stable")
        self.owners = owners[order]
        self.starts = np.searchsorted(kmers[order], np.arange(RADIX ** 3 + 1))

    @staticmethod
    def encode(sequence):
        """The sequence's 3-mers as integers, or nothing for a sequence shorter than three."""
        codes = np.array([INDEX[residue] for residue in sequence], dtype=np.int32)
        if len(codes) < 3:
            return np.empty(0, dtype=np.int32)
        return codes[:-2] * RADIX ** 2 + codes[1:-1] * RADIX + codes[2:]

    def nearest(self, sequence):
        """(index into the reference, identity), or (None, nan) when there is nothing to align.

        The query has to be valid - `is_valid` - because a residue outside the alphabet has no
        code and no alignment. A query shorter than three residues shares no 3-mer with anything,
        so it has no candidates and no answer.
        """
        ids = np.unique(self.encode(sequence))
        if not len(ids):
            return None, float("nan")
        hits = np.concatenate([self.owners[self.starts[i]:self.starts[i + 1]] for i in ids])
        shared = np.bincount(hits, minlength=len(self.reference))
        candidates = np.argsort(-shared)[:self.top_k]
        best = max(candidates, key=lambda c: identity(sequence, self.reference[c]))
        return int(best), identity(sequence, self.reference[best])


def sequence_metrics(samples, reference, *, top_k=DEFAULT_TOP_K, log=None):
    """`samples` with `valid`, `nearest_identity` and `is_unique` added. One row in, one row out.

    `samples` is what `read_sequences` returns and `reference` the list of sequences novelty is
    measured against - `read_reference("data/train.csv")`, what the model was trained on.

    valid             non-empty and made only of the 20 standard residues. Everything a model
                      emits gets a row, so this is the honest denominator for the rest.
    nearest_identity  identity to the closest reference sequence, NaN for an invalid sample. Near
                      1 is copying; low is novel, but a random sequence is novel too, so read it
                      against the real held-out proteins rather than against 0.
    is_unique         whether this sequence appears for the first time in its own set. It is
                      computed within each `setting` group, because a sequence being a duplicate
                      of one in a different set says nothing, and its mean over a setting is the
                      fraction of distinct sequences.
    """
    frame = samples.copy()
    if "setting" not in frame:
        frame["setting"] = "samples"
    sequences = frame["sequence"].tolist()

    frame["valid"] = [is_valid(sequence) for sequence in sequences]

    if log:
        log(f"indexing {len(reference)} reference sequences")
    finder = NearestFinder(list(reference), top_k=top_k)
    nearest = []
    for done, (sequence, valid) in enumerate(zip(sequences, frame["valid"]), start=1):
        nearest.append(finder.nearest(sequence)[1] if valid else np.nan)
        if log and done % 100 == 0:
            log(f"  nearest reference {done}/{len(sequences)}")
    frame["nearest_identity"] = nearest

    frame["is_unique"] = True
    for _, group in frame.groupby("setting", sort=False):
        frame.loc[group.index, "is_unique"] = ~group["sequence"].duplicated().values
    return frame


def set_metrics(sequences, reference, *, k_max=3):
    """The numbers that belong to a set rather than to any one sequence.

    Only the k-mer divergences: a Jensen-Shannon divergence between the k-mer frequencies of a
    whole set and of the reference, in bits, 0 for identical distributions. It is biased upwards
    for small samples, badly so at k = 3, so compare sets of the same size and read it against the
    real-protein control rather than against 0.

    `sequences` should be the valid sequences of one set; everything else a report wants about the
    set is an aggregate of the columns `sequence_metrics` returns.
    """
    sequences = list(sequences)
    reference = list(reference)
    metrics = {"n_sequences": len(sequences), "n_reference": len(reference)}
    for k in range(1, k_max + 1):
        metrics[f"js_divergence_{k}mer"] = js_divergence(kmer_distribution(sequences, k),
                                                         kmer_distribution(reference, k))
    return metrics


def summarize(table):
    """The set-level view of a `sequence_metrics` table, as one row per setting.

    Every number is an aggregate of a column, which is why it is derived on every run rather than
    stored: the table is the measurement and this is a way of reading it.
    """
    rows = {}
    for setting, group in table.groupby("setting", sort=False):
        valid = group[group["valid"]]
        rows[setting] = {
            "n": len(group),
            "frac_valid": float(group["valid"].mean()),
            "frac_unique": float(group["is_unique"].mean()),
            "nearest_identity_median": float(valid["nearest_identity"].median()),
        }
    return pd.DataFrame(rows).T.rename_axis("setting")

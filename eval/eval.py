"""Score a set of generated sequences: three metrics, one row per sequence.

Takes one CSV with a `sequence` column - whatever your sampling code writes, or any other table of
sequences - and writes it back with a column per metric, so the granularity the measurements have
is the granularity on disk. Nothing here is set-level by nature; `summary.csv` is that table read
as aggregates of its columns, which is why it is derived on every run rather than stored as the
result.

Only `sequence` is required. The length is computed from it rather than read, so a `length` column
that disagrees with its sequence cannot quietly skew the report, and ids are made up (`sample_0`,
`sample_1`, ...) when the CSV has none. Any other column is carried through untouched.

The reference novelty is measured against is another such CSV, `data/train.csv` by default and
`--reference` for any other. Reading it the same way means a sequence written in lower case, in
either file, is the same protein to both.

    uv run python eval/eval.py --samples results/sequences/mock_data.csv
    uv run --group eval python eval/eval.py --samples results/sequences/mock_data.csv --fold

writes, into results/eval/mock_data/ unless --out says otherwise,

    eval.csv     one row per sample: its own columns plus the metrics below, so it can be read
                 straight back into this script
    summary.csv  one row per set, every number an aggregate of a column of eval.csv
    pdb/         with --fold: one ESMFold structure per valid sample, pLDDT in the B-factor column

Three metrics, one column each, all of them per sequence:

    valid             is it a protein: non-empty and only the 20 standard residues
    nearest_identity  is it new: identity to the closest sequence of the reference, which is
                      data/train.csv by default - the sequences the model was trained on. Measuring
                      against the whole data set instead would count a generated copy of a
                      held-out protein as novelty rather than as memorisation. NaN for an invalid
                      sample.
    is_unique         is it the same one again: whether the sequence appears for the first time in
                      its own set

and with --fold a fourth, `mean_plddt`, ESMFold's own confidence in the structure it predicts,
averaged over the residues.

Novelty needs a yardstick, because a random sequence is perfectly novel and a memorised one scores
best on everything else. Real held-out proteins are that yardstick: write them with
`eval/make_controls.py` and score them through this same script.

`protgen.metrics.sequences` has the pieces to add a column when one of these raises a question it
cannot answer - sequence entropy, homopolymer runs, k-mer divergence from the reference, mean
identity to the other samples.

One CSV in, one directory out. A CSV that already carries a `setting` column - several sampling
temperatures concatenated into one file, say - is grouped by it instead: `is_unique` then compares
a sample only with the rest of its own group, and summary.csv has a row per group. Otherwise the
whole file is one set, named after itself, and eval.csv leaves the column out, because the output
directory already says it.
"""

import argparse
from pathlib import Path

import pandas as pd

from protgen.metrics import (
    DEFAULT_TOP_K,
    read_reference,
    read_sequences,
    sequence_metrics,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent

# The sequences a generated one is called novel or memorised against: the training split, because
# that is what the model saw. data/val.csv or data/test.csv answer a different question.
DEFAULT_REFERENCE = ROOT / "data/train.csv"

# Decimals kept for nearest_identity in eval.csv. It is a ratio of two integers over an alignment
# of at most a few hundred positions, so five decimals is more than the quantity can mean; the
# rest is float noise that makes the column unreadable in a terminal or a spreadsheet.
IDENTITY_DECIMALS = 5


def shorten(path):
    """The path relative to the project root, which is how the README and the docstrings write it.

    The defaults here are absolute, because they are built from this file's location rather than
    from the current directory, and an absolute path on every line of output is noise. A path
    outside the project - an --out somewhere else - has no shorter form and is printed as it is.
    """
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def for_reading(table):
    """The table as eval.csv stores it: no redundant column, rounded, sequence last.

    One CSV in, one output directory named after it, so a `setting` of one value is the path said
    twice and is dropped. A file holding several sets keeps it, because then it is the only thing
    that says which set a row belongs to. `summary.csv` is indexed by it either way.

    The sequence goes last because it is a hundred characters wide and every column after it would
    be pushed off the screen; it stays in the file because a row that cannot be read back to the
    sequence it describes is hard to check.
    """
    if table["setting"].nunique() == 1:
        table = table.drop(columns="setting")
    table = table.round({"nearest_identity": IDENTITY_DECIMALS})
    return table[[name for name in table.columns if name != "sequence"] + ["sequence"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True,
                        help="a CSV with a `sequence` column, e.g. results/sequences/mock_data.csv")
    parser.add_argument("--out", help="output directory (default: results/eval/<csv name>)")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                        help="CSV of the sequences novelty is measured against "
                             "(default: data/train.csv, what the model was trained on)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"reference candidates aligned per sample (default: {DEFAULT_TOP_K})")
    parser.add_argument("--limit", type=int, help="only score the first N rows")
    parser.add_argument("--fold", action="store_true",
                        help="also fold every valid sample with ESMFold (needs a GPU and the eval "
                             "dependency group)")
    parser.add_argument("--cpu-only", action="store_true", help="fold on the CPU; very slow")
    parser.add_argument("--max-tokens-per-batch", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="ESMFold axial attention chunk size (lower = less GPU memory)")
    parser.add_argument("--num-recycles", type=int, help="ESMFold recycles (default: its own, 4)")
    args = parser.parse_args()

    # A missing column or an unusable reference is a mistake in the input, not a bug: report it as
    # a message and an exit code rather than as a traceback.
    try:
        samples = read_sequences(args.samples)
        reference = read_reference(args.reference)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))

    if args.limit:
        samples = samples.head(args.limit)
    out = Path(args.out) if args.out else ROOT / "results/eval" / Path(args.samples).stem
    sets = samples["setting"].nunique()
    print(f"{len(samples)} sequences{f' in {sets} sets' if sets > 1 else ''}, against "
          f"{len(reference)} in {shorten(args.reference)}")

    table = sequence_metrics(samples, reference, top_k=args.top_k,
                             log=lambda message: print(message, flush=True))

    if args.fold:
        from protgen.metrics.fold import fold, pick_device

        valid = table[table["valid"]]
        rows = fold(zip(valid["id"], valid["sequence"]), out / "pdb",
                    device=pick_device(args.cpu_only), max_tokens=args.max_tokens_per_batch,
                    chunk_size=args.chunk_size, num_recycles=args.num_recycles,
                    log=lambda message: print(message, flush=True))
        # An invalid sample keeps its row and gets no structure, as with every other metric here.
        table = table.merge(pd.DataFrame(rows), on="id", how="left")

    out.mkdir(parents=True, exist_ok=True)
    written = for_reading(table)
    written.to_csv(out / "eval.csv", index=False)

    summary = summarize(table)
    if args.fold:
        for setting, group in table.groupby("setting", sort=False):
            summary.loc[setting, "mean_plddt"] = group["mean_plddt"].mean()
    summary.to_csv(out / "summary.csv")

    print("\n" + summary.T.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {shorten(out / 'eval.csv')} ({len(written)} rows, "
          f"{', '.join(written.columns)}) and {shorten(out / 'summary.csv')}")
    if args.fold:
        print(f"Structures in {shorten(out / 'pdb')}")


if __name__ == "__main__":
    main()

"""Write the two control sample sets that make the metrics readable.

A single number from eval.py says very little on its own: real proteins do not reach 0 on the
k-mer divergences either, and a random sequence is perfectly novel. Two controls fix the scale.

positive  control_real.csv      real held-out proteins (the test split) that no model trained on.
                                They are what the metrics look like for sequences that are
                                certainly proteins, and the yardstick for novelty: a sample set
                                whose identity to the nearest training sequence sits *above*
                                theirs is reproducing training data.
negative  control_shuffled.csv  the same proteins with their residues shuffled. Length and
                                composition are untouched, so whatever separates the two controls
                                comes from the order of the residues and nothing else.

They are written as ordinary sample CSVs, next to the generated ones, so eval.py scores them
through exactly the same code path - the yardstick is a row in the comparison rather than a
special case in the report.

Use the same --num as the number of generated sequences: the k-mer divergences and the diversity
metrics depend on the size of the set.

Example:
    uv run python eval/make_controls.py --num 256
    uv run python eval/eval.py --samples results/sequences/control_real.csv
    uv run python eval/eval.py --samples results/sequences/control_shuffled.csv
"""

import argparse
from pathlib import Path

from protgen.metrics import controls, read_reference, write_sequences

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "results/sequences"),
                        help="directory for the two CSVs (default: results/sequences, beside the "
                             "generated samples)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test",
                        help="the split the controls are drawn from (default: test, the held-out "
                             "proteins no model trained on)")
    parser.add_argument("--num", type=int, default=256,
                        help="proteins in each control; match the number of generated samples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split = ROOT / "data" / f"{args.split}.csv"
    try:
        sequences = read_reference(split)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))

    real, shuffled = controls(sequences, n=args.num, seed=args.seed)
    out = Path(args.out)
    for name, rows in (("control_real", real), ("control_shuffled", shuffled)):
        path = write_sequences(out / f"{name}.csv", rows)
        print(f"wrote {len(rows)} sequences to {path}")


if __name__ == "__main__":
    main()

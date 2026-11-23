import argparse

import pandas as pd

from protgen.metrics.sequences import NearestFinder, is_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="CSV with sequence, target, setting columns")
    parser.add_argument("--reference", default="data/train.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    reference = pd.read_csv(args.reference)
    finder = NearestFinder(reference["sequence"].tolist())

    families, identities = [], []
    for sequence in samples["sequence"]:
        if is_valid(sequence):
            index, identity = finder.nearest(sequence)
            families.append(reference["family"].iloc[index])
            identities.append(identity)
        else:
            families.append(None)
            identities.append(float("nan"))
    samples["neighbour_family"] = families
    samples["nearest_identity"] = identities
    samples["neighbour_is_target"] = samples["neighbour_family"] == samples["target"]

    # classifier guidance writes gamma + predicted_by_guiding_head; classifier-free guidance writes w + predicted_by_judge
    strength = "w" if "w" in samples else "gamma"
    verdict = "predicted_by_judge" if "predicted_by_judge" in samples else "predicted_by_guiding_head"

    summary = samples.groupby("setting", sort=False).agg(
        strength=(strength, "first"),
        n=("sequence", "size"),
        head_says_target=(verdict, "mean"),
        neighbour_is_target=("neighbour_is_target", "mean"),
        median_nearest_identity=("nearest_identity", "median"),
        mean_length=("sequence", lambda s: s.str.len().mean()),
        unique=("sequence", lambda s: s.nunique() / len(s)),
    )
    print(summary.round(3).to_string())
    out = args.out or args.samples.replace(".csv", "_family_check.csv")
    summary.to_csv(out)
    samples.to_csv(out.replace(".csv", "_per_sample.csv"), index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

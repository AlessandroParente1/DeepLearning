import random


def controls(sequences, n=500, seed=0):
    """(real, shuffled): two lists of sample rows drawn from `sequences`.

    `sequences` is a list of residue strings, normally the test split - proteins no model was
    trained on - as `read_reference("data/test.csv")` returns them.

    real      `n` of those proteins, unchanged: what every metric looks like for sequences that
              certainly are proteins.
    shuffled  the same proteins with their residues permuted. Length and composition are
              untouched, so whatever separates the two sets comes from the order of the residues
              and nothing else.

    `n` should match the number of generated sequences being compared, because some of the metrics
    depend on the size of the set. The rows carry only `id` and `sequence`, the shape a generated
    sample CSV has; everything else is derived from the sequence when the set is scored.
    """
    rng = random.Random(seed)
    chosen = rng.sample(list(sequences), min(n, len(sequences)))

    real = [{"id": f"real_{i}", "sequence": sequence}
            for i, sequence in enumerate(chosen)]
    shuffled = [{"id": f"shuffled_{i}", "sequence": "".join(rng.sample(sequence, len(sequence)))}
                for i, sequence in enumerate(chosen)]
    return real, shuffled

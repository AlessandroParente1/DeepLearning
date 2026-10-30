from pathlib import Path
import time

import numpy as np
import torch

from protgen.metrics.sequences import is_valid

MODEL_NAME = "facebook/esmfold_v1"
# batch_size * longest_sequence_in_batch, the quantity that drives peak memory. 1024 fits a 16 GB
# V100 for proteins of this length.
DEFAULT_MAX_TOKENS = 1024
# Axial attention chunk size. Lower means less memory and slower folding.
DEFAULT_CHUNK_SIZE = 64


def pick_device(cpu_only=False):
    """cuda when there is one, unless `cpu_only`. CPU works but is slow enough to hurt."""
    return torch.device("cpu" if cpu_only or not torch.cuda.is_available() else "cuda")


def load_model(device, chunk_size=DEFAULT_CHUNK_SIZE, num_recycles=None):
    """(tokenizer, model), in eval mode on `device`. Downloads ~8 GB the first time."""
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
    model = model.to(device)
    if device.type == "cuda":
        model.esm = model.esm.half()  # the language model stem is the memory hog
    model.trunk.set_chunk_size(chunk_size)
    model.eval()
    if num_recycles is not None:
        model.config.esmfold_config.trunk.max_recycles = num_recycles
    return tokenizer, model


def make_batches(records, max_tokens=DEFAULT_MAX_TOKENS):
    """Group (id, sequence) records so that batch_size * longest_length <= max_tokens.

    Sorted by length first, so that a batch is mostly one length and little of the padded work is
    wasted.
    """
    batch = []
    for record in sorted(records, key=lambda record: len(record[1])):
        if batch and (len(batch) + 1) * len(record[1]) > max_tokens:
            yield batch
            batch = []
        batch.append(record)
    if batch:
        yield batch


def outputs_to_pdbs(outputs, lengths):
    """(pdb strings, per-residue pLDDT) from one batch of ESMFold outputs.

    The batch is padded to its longest sequence, so everything is cut back to each sequence's own
    length. The model's `ptm` output is one value for the whole batch, so it is not reported.
    """
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
    from transformers.models.esm.openfold_utils.protein import to_pdb

    positions = atom14_to_atom37(outputs["positions"][-1], outputs).cpu().numpy()
    arrays = {key: value.cpu().numpy() for key, value in outputs.items() if torch.is_tensor(value)}
    plddt = arrays["plddt"].astype(np.float32)  # (batch, length, 37)
    if plddt.max() <= 1.0:  # some versions return a fraction rather than 0-100
        plddt = plddt * 100

    pdbs, per_residue = [], []
    for i, length in enumerate(lengths):
        protein = OFProtein(
            aatype=arrays["aatype"][i, :length],
            atom_positions=positions[i, :length],
            atom_mask=arrays["atom37_atom_exists"][i, :length],
            residue_index=arrays["residue_index"][i, :length] + 1,
            b_factors=plddt[i, :length],
        )
        pdbs.append(to_pdb(protein))
        per_residue.append(plddt[i, :length, 1])  # the CA atom: one value per residue
    return pdbs, per_residue


def fold(records, pdb_dir, *, device=None, max_tokens=DEFAULT_MAX_TOKENS,
         chunk_size=DEFAULT_CHUNK_SIZE, num_recycles=None, log=None):
    """Fold (id, sequence) records into `pdb_dir`; one row per sequence, keyed on `id`.

    Every record must be a valid sequence of standard residues: ESMFold has no token for anything
    else, so an invalid sequence is refused here rather than folded into nonsense. Filter the
    sample table on its `valid` column first.

    The rows carry only `mean_plddt` and `pdb` beside the id, because the sequence and its length
    are already in the sample table these join onto.
    """
    records = [(str(name), sequence) for name, sequence in records]
    invalid = [name for name, sequence in records if not is_valid(sequence)]
    if invalid:
        shown = ", ".join(invalid[:5]) + (" ..." if len(invalid) > 5 else "")
        raise ValueError(f"{len(invalid)} sequence(s) are empty or hold letters outside the 20 "
                         f"standard residues, which ESMFold cannot fold: {shown}")
    if not records:
        raise ValueError("No sequences to fold")

    device = device or pick_device()
    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"folding {len(records)} sequences on {device}")
        if device.type == "cpu":
            log("  warning: on CPU this is very slow; submit it to a GPU queue for a real run")
    tokenizer, model = load_model(device, chunk_size, num_recycles)

    rows, start = [], time.time()
    for batch in make_batches(records, max_tokens):
        names = [name for name, _ in batch]
        sequences = [sequence for _, sequence in batch]
        tokens = tokenizer(sequences, return_tensors="pt", add_special_tokens=False, padding=True)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            outputs = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])
        pdbs, per_residue = outputs_to_pdbs(outputs, [len(s) for s in sequences])
        for name, structure, plddt in zip(names, pdbs, per_residue):
            path = pdb_dir / f"{name}.pdb"
            path.write_text(structure)
            rows.append({"id": name, "mean_plddt": float(plddt.mean()), "pdb": str(path)})
        if log:
            log(f"  folded {len(rows)}/{len(records)}  ({time.time() - start:.0f}s)")
    return rows

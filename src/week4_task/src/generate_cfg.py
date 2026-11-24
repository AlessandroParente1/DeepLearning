"""Classifier-free guidance:  log p~(x_i | x_<i, y) = (1 + w) log p(x_i | x_<i, y) - w log p(x_i | x_<i, null)     (Eq. 12)

    uv run python src/generate_cfg.py --target HU --ws 0 0.5 1 2 4 --n 64

One conditional model gives both terms: the same network is run once with the family label and once with the "null" label
(it was trained with label dropout, so it can do both). w = 0 is plain conditional sampling; larger w pushes further in the
direction the label moves the prediction. (w = -1 would be fully unconditional, since the first term then vanishes.)

Cost: 2 forward passes per step, against 21 predictor evaluations per step for classifier guidance (generate_guided.py).

The samples are judged by the family head of the ORIGINAL small_b (checkpoints/family_head_small_b.pt). That judge took no part in
generating these samples, unlike in classifier guidance where the guiding head is also the judge.
"""
import argparse
import csv
import os

import torch
import torch.nn.functional as F

from protgen.data import FAMILY_TO_INDEX, MAX_DATA_LENGTH
from protgen.models.classifier_head import FamilyHead
from generate_guided import BOS, CANDIDATES, EOS, INDEX_TO_TOKEN, PAD, predicted_family
from train_classifier_head import DEVICE, load_backbone, set_seed


@torch.no_grad()
def generate_batch(model, target: int, w: float, batch_size: int, temperature: float, generator):
    candidates = CANDIDATES.to(DEVICE)  # <EOS> + the 20 residues, never <PAD> / <BOS>
    labels = torch.cat([torch.full((batch_size,), target, device=DEVICE),
                        torch.full((batch_size,), model.num_labels, device=DEVICE)])  # first half: family, second half: null
    tokens = torch.full((batch_size, MAX_DATA_LENGTH), PAD, dtype=torch.long, device=DEVICE)
    tokens[:, 0] = BOS
    done = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    for i in range(MAX_DATA_LENGTH - 1):
        prefix = tokens[:, : i + 1]
        logits = model(prefix.repeat(2, 1), labels)[:, i][:, candidates]  # one pass for conditional AND unconditional
        conditional, unconditional = logits.chunk(2)
        log_conditional = F.log_softmax(conditional / temperature, dim=-1)
        log_unconditional = F.log_softmax(unconditional / temperature, dim=-1)
        score = (1 + w) * log_conditional - w * log_unconditional
        if i == MAX_DATA_LENGTH - 2:  # out of room: must end here
            score[:, 1:] = float("-inf")

        choice = torch.multinomial(F.softmax(score, dim=-1), 1, generator=generator).squeeze(1)
        next_token = torch.where(done, torch.full_like(choice, PAD), candidates[choice])
        tokens[:, i + 1] = next_token
        done = done | (next_token == EOS)
        if done.all():
            break

    sequences = []
    for row in tokens.tolist():
        residues = []
        for t in row[1:]:
            if t == EOS or t == PAD:
                break
            residues.append(INDEX_TO_TOKEN[t])
        sequences.append("".join(residues))
    return sequences, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/small_b_cfg.pt", help="conditional transformer (train_transformer.py --conditional)")
    parser.add_argument("--judge-head", default="checkpoints/family_head_small_b.pt",
                        help="family head (and its backbone) used only to judge the samples")
    parser.add_argument("--target", default="HU", choices=list(FAMILY_TO_INDEX))
    parser.add_argument("--ws", type=float, nargs="+", default=[0, 0.5, 1, 2, 4])
    parser.add_argument("--n", type=int, default=64, help="sequences per w")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="CSV, default results/sequences/cfg_<target>.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
    model = load_backbone(args.model)
    assert model.num_labels > 0, f"{args.model} is not a conditional model; train it with --conditional"
    judge_checkpoint = torch.load(args.judge_head, weights_only=True, map_location=DEVICE)
    judge_backbone = load_backbone(judge_checkpoint["backbone"])
    judge_head = FamilyHead(judge_backbone.config["d_model"]).to(DEVICE)
    judge_head.load_state_dict(judge_checkpoint["head"])
    judge_head.eval()
    target = FAMILY_TO_INDEX[args.target]

    out = args.out or f"results/sequences/cfg_{args.target}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for w in args.ws:
        sequences, verdicts = [], []
        for start in range(0, args.n, args.batch_size):
            batch = min(args.batch_size, args.n - start)
            seqs, tokens = generate_batch(model, target, w, batch, args.temperature, generator)
            sequences += seqs
            verdicts += predicted_family(judge_backbone, judge_head, tokens).tolist()
        share = sum(v == target for v in verdicts) / len(verdicts)
        mean_len = sum(map(len, sequences)) / len(sequences)
        print(f"w {w:>5}: judge says {args.target} for {share:.2%}, mean length {mean_len:.1f}, "
              f"unique {len(set(sequences))}/{len(sequences)}")
        for s, v in zip(sequences, verdicts):
            rows.append({"setting": f"{args.target}_w{w:g}", "w": w, "target": args.target,
                         "predicted_by_judge": v == target, "sequence": s})

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} sequences to {out}")


if __name__ == "__main__":
    main()

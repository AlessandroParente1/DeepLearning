import argparse
import csv
import os

import torch
import torch.nn.functional as F

from protgen.data import ALL_TOKENS, FAMILY_TO_INDEX, MAX_DATA_LENGTH, VOCABULARY, ProteinDataset
from protgen.models.classifier_head import FamilyHead
from train_classifier_head import DEVICE, load_backbone, set_seed

EOS, PAD, BOS = VOCABULARY["<EOS>"], VOCABULARY["<PAD>"], VOCABULARY["<BOS>"]
INDEX_TO_TOKEN = {i: t for t, i in VOCABULARY.items()}
CANDIDATES = torch.tensor([EOS] + [VOCABULARY[a] for a in ALL_TOKENS[3:]])  # <EOS> + 20 residues (never <PAD>/<BOS>)


@torch.no_grad()
def generate_batch(backbone, head, target: int, gamma: float, batch_size: int, temperature: float, generator):
    candidates = CANDIDATES.to(DEVICE)
    num_candidates = len(candidates)
    tokens = torch.full((batch_size, MAX_DATA_LENGTH), PAD, dtype=torch.long, device=DEVICE)
    tokens[:, 0] = BOS
    done = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    # token index i + 1 is chosen at step i; the longest protein has 127 residues, so the last step (i = 127) writes <EOS>
    for i in range(MAX_DATA_LENGTH - 1):
        prefix = tokens[:, : i + 1]
        copies = prefix[:, None, :].expand(batch_size, num_candidates, i + 1)
        trial = torch.cat([copies, candidates.view(1, -1, 1).expand(batch_size, -1, 1)], dim=2)  # prefix + candidate
        features = backbone.features(trial.reshape(batch_size * num_candidates, i + 2))
        features = features.view(batch_size, num_candidates, i + 2, -1)

        # the next-token distribution only depends on the prefix, so any candidate's copy gives the same logits at i
        generative_logits = backbone.to_logits(features[:, 0, i])[:, candidates]  # BATCH x CANDIDATES
        score = F.log_softmax(generative_logits / temperature, dim=-1)
        if gamma != 0:
            # p(y | x_<i, a): the head on the hidden state at the candidate's own position
            family_log_prob = F.log_softmax(head(features[:, :, i + 1]), dim=-1)[..., target]  # BATCH x CANDIDATES
            score = score + gamma * family_log_prob
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


@torch.no_grad()
def predicted_family(backbone, head, tokens):
    lengths = ((tokens != PAD) & (tokens != EOS)).sum(1) - 1  # residues (without <BOS>)
    ends = (tokens == EOS).float().argmax(1)  # index of <EOS>
    features = backbone.features(tokens)
    return head(features[torch.arange(len(tokens)), ends]).argmax(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="HU", choices=list(FAMILY_TO_INDEX))
    parser.add_argument("--gammas", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--n", type=int, default=64, help="sequences per gamma")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--head", default="checkpoints/family_head.pt")
    parser.add_argument("--out", default=None, help="CSV, default results/sequences/guided_<target>.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
    checkpoint = torch.load(args.head, weights_only=True, map_location=DEVICE)
    backbone = load_backbone(checkpoint["backbone"])
    head = FamilyHead(backbone.config["d_model"]).to(DEVICE)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    target = FAMILY_TO_INDEX[args.target]

    out = args.out or f"results/sequences/guided_{args.target}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for gamma in args.gammas:
        sequences, verdicts = [], []
        for start in range(0, args.n, args.batch_size):
            batch = min(args.batch_size, args.n - start)
            seqs, tokens = generate_batch(backbone, head, target, gamma, batch, args.temperature, generator)
            sequences += seqs
            verdicts += predicted_family(backbone, head, tokens).tolist()
        share = sum(v == target for v in verdicts) / len(verdicts)
        mean_len = sum(map(len, sequences)) / len(sequences)
        print(f"gamma {gamma:>5}: head says {args.target} for {share:.2%}, mean length {mean_len:.1f}, "
              f"unique {len(set(sequences))}/{len(sequences)}")
        for s, v in zip(sequences, verdicts):
            rows.append({"setting": f"{args.target}_gamma{gamma:g}", "gamma": gamma, "target": args.target,
                         "predicted_by_guiding_head": v == target, "sequence": s})

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} sequences to {out}")


if __name__ == "__main__":
    main()

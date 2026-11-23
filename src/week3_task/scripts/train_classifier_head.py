import argparse
import copy
import csv
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import FAMILIES, ProteinDataset
from protgen.models.classifier_head import FamilyHead, position_mask
from protgen.models.transformer import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIXES_TO_REPORT = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone(path: str):
    saved = torch.load(path, weights_only=True, map_location=DEVICE)
    # new checkpoints are {"config": ..., "state": ...}; the original big model is a bare state dict
    config, state = (saved["config"], saved["state"]) if "state" in saved else ({}, saved)
    backbone = Transformer(**config).to(DEVICE)
    backbone.load_state_dict(state)
    backbone.eval()  # dropout off, always
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


def head_loss_and_logits(backbone, head, batch):
    x, _, y, length = (t.to(DEVICE) for t in batch)
    with torch.no_grad():
        features = backbone.features(x)
    logits = head(features)  # BATCH x POSITIONS x 3
    mask = position_mask(length, x.shape[1])
    targets = y[:, None].expand_as(mask)
    return logits, targets, mask


def train_one_epoch(backbone, head, loader, optimizer):
    head.train()
    total = 0.0
    for batch in loader:
        logits, targets, mask = head_loss_and_logits(backbone, head, batch)
        loss = F.cross_entropy(logits[mask], targets[mask])  # mean over every (protein, prefix) pair
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(backbone, head, loader):
    head.eval()
    max_positions = 129
    correct = torch.zeros(max_positions)
    count = torch.zeros(max_positions)
    nll = torch.zeros(max_positions)
    for batch in loader:
        logits, targets, mask = head_loss_and_logits(backbone, head, batch)
        token_nll = F.cross_entropy(logits.permute(0, 2, 1), targets, reduction="none")  # BATCH x POSITIONS
        hit = (logits.argmax(-1) == targets).float()
        T = mask.shape[1]
        correct[:T] += (hit * mask).sum(0).cpu()
        nll[:T] += (token_nll * mask).sum(0).cpu()
        count[:T] += mask.sum(0).cpu()
    overall_loss = (nll.sum() / count.sum()).item()
    overall_accuracy = (correct.sum() / count.sum()).item()
    per_position = {"correct": correct, "count": count, "nll": nll}
    return overall_loss, overall_accuracy, per_position


@torch.no_grad()
def complete_sequence_accuracy(backbone, head, loader):
    head.eval()
    right, total = 0, 0
    per_family = torch.zeros(len(FAMILIES), len(FAMILIES), dtype=torch.long)
    for batch in loader:
        x, _, y, length = (t.to(DEVICE) for t in batch)
        features = backbone.features(x)
        last = features[torch.arange(len(x)), length + 1]
        prediction = head(last).argmax(-1)
        right += (prediction == y).sum().item()
        total += len(y)
        for true, pred in zip(y.tolist(), prediction.tolist()):
            per_family[true, pred] += 1
    return right / total, per_family


def save_curve(per_position, path_csv, path_png, title):
    positions = torch.arange(len(per_position["count"]))
    keep = per_position["count"] > 50  # skip positions only a handful of proteins reach
    rows = []
    for p in positions[keep].tolist():
        n = per_position["count"][p].item()
        rows.append((p, int(n), per_position["correct"][p].item() / n, per_position["nll"][p].item() / n))
    with open(path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_length", "n", "accuracy", "nll"])
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], lw=1.5)
    ax.axhline(1 / len(FAMILIES), color="gray", ls="--", lw=1, label="chance (1/3)")
    ax.set_xlabel("number of residues seen")
    ax.set_ylabel("family accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="transformer_1dot5one_weights_1794778911.2550974.pt")
    parser.add_argument("--out", default="checkpoints/family_head.pt")
    parser.add_argument("--results", default="results/classifier")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    backbone = load_backbone(args.backbone)
    head = FamilyHead(backbone.config["d_model"]).to(DEVICE)
    print(f"head parameters: {sum(p.numel() for p in head.parameters()):,}  (backbone frozen)")

    train_loader = DataLoader(ProteinDataset("data/train.csv"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ProteinDataset("data/val.csv"), batch_size=args.batch_size)
    test_loader = DataLoader(ProteinDataset("data/test.csv"), batch_size=args.batch_size)

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    history = []
    for epoch in range(args.max_epochs):
        train_loss = train_one_epoch(backbone, head, train_loader, optimizer)
        val_loss, val_acc, _ = evaluate(backbone, head, val_loader)
        history.append((epoch + 1, train_loss, val_loss, val_acc))
        print(f"epoch {epoch + 1}: train loss {train_loss:.4f}  val loss {val_loss:.4f}  val acc (all prefixes) {val_acc:.4f}")
        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(head.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                print(f"Stopping after {epoch + 1} epochs.")
                break

    head.load_state_dict(best_state)
    torch.save({"head": best_state, "backbone": args.backbone, "seed": args.seed}, args.out)

    with open(os.path.join(args.results, "head_training_history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc_all_prefixes"])
        writer.writerows(history)

    # validation curve for model selection, test numbers once at the end
    for name, loader in (("val", val_loader), ("test", test_loader)):
        loss, acc, per_position = evaluate(backbone, head, loader)
        complete_acc, confusion = complete_sequence_accuracy(backbone, head, loader)
        rows = save_curve(per_position, os.path.join(args.results, f"accuracy_by_prefix_{name}.csv"),
                          os.path.join(args.results, f"accuracy_by_prefix_{name}.png"),
                          f"Noisy predictor, {name} set")
        print(f"\n[{name}] loss over all prefixes {loss:.4f}, accuracy over all prefixes {acc:.4f}, "
              f"accuracy on complete proteins {complete_acc:.4f}")
        print(f"[{name}] confusion (rows = true {FAMILIES}, columns = predicted):\n{confusion.numpy()}")
        by_prefix = {r[0]: r[2] for r in rows}
        print(f"[{name}] accuracy by number of residues seen: "
              + ", ".join(f"{p}: {by_prefix[p]:.3f}" for p in PREFIXES_TO_REPORT if p in by_prefix))


if __name__ == "__main__":
    main()

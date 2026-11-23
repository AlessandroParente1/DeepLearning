import argparse
import copy
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from protgen.data.make_data import FAMILIES, IGNORE_INDEX, VOCABULARY_SIZE, ProteinDataset
from protgen.models.transformer import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="transformer")
    parser.add_argument("--conditional", action="store_true", help="family label as an extra input (classifier-free guidance)")
    parser.add_argument("--p-drop", type=float, default=0.15, help="chance of replacing the label by 'null' while training")
    args = parser.parse_args()

    set_seed(args.seed)
    config = dict(d_model=args.d_model, heads=args.heads, blocks=args.blocks, ffn=args.ffn, dropout=args.dropout,
                  num_labels=len(FAMILIES) if args.conditional else 0)
    model = Transformer(**config).to(DEVICE)
    print(f"{args.name}: {sum(p.numel() for p in model.parameters()):,} parameters, config {config}")
    p_drop = args.p_drop if args.conditional else 0.0

    train_loader = DataLoader(ProteinDataset("data/train.csv"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ProteinDataset("data/val.csv"), batch_size=args.batch_size)
    test_loader = DataLoader(ProteinDataset("data/test.csv"), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(args.max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, p_drop)
        val_loss = evaluate(model, val_loader, use_labels=True)  # with the true family (or the only loss, if unconditional)
        val_null = evaluate(model, val_loader, use_labels=False) if args.conditional else float("nan")
        score = (val_loss + val_null) / 2 if args.conditional else val_loss  # a CFG model has to be good at both
        history.append((epoch + 1, train_loss, val_loss, val_null))
        extra = f", validation loss with null label {val_null:.4f}" if args.conditional else ""
        print(f"epoch {epoch + 1}: training loss {train_loss:.4f},  validation loss {val_loss:.4f}{extra}")
        if score < best_score:
            best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad > args.patience:
                print(f"Stopping after {epoch + 1} epochs.")
                break

    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, use_labels=True)
    test_null = evaluate(model, test_loader, use_labels=False) if args.conditional else float("nan")
    print(f"best validation score {best_score:.4f}, test loss {test_loss:.4f}"
          + (f", test loss with null label {test_null:.4f}" if args.conditional else ""))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/lm", exist_ok=True)
    torch.save({"config": config, "state": best_state}, f"checkpoints/{args.name}.pt")
    with open(f"results/lm/{args.name}_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_loss_null_label"])
        writer.writerows(history)
        writer.writerow(["test", "", test_loss, test_null])


def forward_with_labels(model: nn.Module, x, y, use_labels: bool):
    """Plain forward for an unconditional model; for a conditional one, the true labels or the null label for everyone."""
    if model.num_labels > 0 and use_labels:
        return model(x, y)
    return model(x)  # unconditional model, or a conditional one asked for its null-label (unconditional) prediction


def train_one_epoch(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, p_drop: float = 0.0):
    model.train()
    total = 0.0
    for x, x_shifted, y, _ in data_loader:
        x, x_shifted, y = x.to(DEVICE), x_shifted.to(DEVICE), y.to(DEVICE)
        if model.num_labels > 0:
            # label dropout: each protein loses its label with probability p_drop, so the same network also learns p(x | no label)
            y = torch.where(torch.rand(len(y), device=DEVICE) < p_drop, torch.full_like(y, model.num_labels), y)
            logits = model(x, y)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCABULARY_SIZE), x_shifted.reshape(-1),
                               ignore_index=IGNORE_INDEX, reduction="mean")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(data_loader)


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, use_labels: bool = True):
    model.eval()
    loss_sum, token_count = 0.0, 0
    for x, x_shifted, y, _ in data_loader:
        x, x_shifted, y = x.to(DEVICE), x_shifted.to(DEVICE), y.to(DEVICE)
        logits = forward_with_labels(model, x, y, use_labels)
        target = x_shifted.reshape(-1)
        loss_sum += F.cross_entropy(logits.reshape(-1, VOCABULARY_SIZE), target,
                                    ignore_index=IGNORE_INDEX, reduction="sum").item()
        token_count += (target != IGNORE_INDEX).sum().item()
    return loss_sum / token_count


if __name__ == "__main__":
    main()

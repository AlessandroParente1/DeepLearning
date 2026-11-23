import torch
from torch import nn

from protgen import data
from protgen.models.transformer import MODEL_DIMENSIONS

HEAD_HIDDEN_SIZE = 256


class FamilyHead(nn.Module):


    def __init__(self, d_model=MODEL_DIMENSIONS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, HEAD_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HEAD_HIDDEN_SIZE, len(data.FAMILIES)),
        )

    # features is BATCH_SIZE x SOME_LENGTH x MODEL_DIMENSIONS, output is BATCH_SIZE x SOME_LENGTH x NUMBER_OF_FAMILIES (logits)
    def forward(self, features):
        return self.layers(features)


def position_mask(lengths: torch.Tensor, total_positions: int):

    positions = torch.arange(total_positions, device=lengths.device)
    return positions[None, :] <= (lengths[:, None] + 1)

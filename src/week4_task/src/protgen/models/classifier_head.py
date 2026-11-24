import torch
from torch import nn

from protgen import data
from protgen.models.transformer import MODEL_DIMENSIONS

HEAD_HIDDEN_SIZE = 256


class FamilyHead(nn.Module):
    """Predicts the protein family p(y | x_<=i) from the frozen transformer's hidden state at position i.

    Because the transformer is causal, the hidden state at position i has only seen <BOS> and the first i residues,
    so the same head works as a "noisy predictor" on any prefix of a sequence.
    """

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
    """True where the head should be trained / evaluated.

    Position p of the input (<BOS>, r1, ..., rL, <EOS>, <PAD>...) has seen p residues, so p = 0 is "nothing yet"
    and p = L + 1 is the finished protein (the <EOS> token has been fed in). Everything after that is padding.
    """
    positions = torch.arange(total_positions, device=lengths.device)
    return positions[None, :] <= (lengths[:, None] + 1)

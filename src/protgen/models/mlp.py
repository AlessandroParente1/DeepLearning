import torch
from torch import nn
from protgen import data

CONTEXT_SIZE = 32
HIDDEN_LAYER_SIZE = 128 # picked some number higher than input dimension and lower than 23^context*_size, maybe use two layers who knows

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # adding + 1 input slot to include index position
            nn.Linear(CONTEXT_SIZE * data.VOCABULARY_SIZE, HIDDEN_LAYER_SIZE), 
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE), 
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, data.VOCABULARY_SIZE),
        )

    # kept as single-argument tensor that is flattened to work with variable context size
    # make sure to pass along the standardised position as well once we add that
    def forward(self, input_tensor):
        input_tensor = self.flatten(input_tensor)
        logits = self.linear_relu_stack(input_tensor)
        return logits
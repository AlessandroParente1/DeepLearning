import torch
from torch import nn
from protgen import data

CONTEXT_SIZE = 3
HIDDEN_LAYER_SIZE1 = 128 
#HIDDEN_LAYER_SIZE2 = 64 

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # adding + 1 input slot later to include index position
            nn.Linear(CONTEXT_SIZE * data.VOCABULARY_SIZE, HIDDEN_LAYER_SIZE1), 
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE1, data.VOCABULARY_SIZE),
        )

    # kept as single-argument tensor that is flattened to work with variable context size
    # make sure to pass along the standardised position as well once we add that
    def forward(self, input_tensor):
        input_tensor = self.flatten(input_tensor)
        logits = self.linear_relu_stack(input_tensor)
        return logits
import torch

from protgen import data

# This is just mean 0 sd 1 normal distribution values on all possible logits
# We are guessing randomly with a varying level of non-zero overconfidence
#
# Included only for comparison since the introduction notebook used randn
def standard_normal_distributed_logits(batch_size: int):
    tensor = torch.randn(batch_size, data.VOCABULARY_SIZE, data.MAX_DATA_LENGTH)
    # -inf used to mask off logits because of softmax definition
    # e^-inf is 0
    tensor[:, data.VOCABULARY["<PAD>"], :] = float("-inf")
    tensor[:, data.VOCABULARY["<BOS>"], :] = float("-inf")
    return tensor

# Here all possible logits are 0 (pre-softmax of course) so they all get same probability
# Essentially we are saying we know nothing
#
# This is the baseline we should be benchmarking against
def uniform_logits(batch_size: int):
    tensor = torch.zeros(batch_size, data.VOCABULARY_SIZE, data.MAX_DATA_LENGTH)
    # -inf used to mask off logits because of softmax definition
    # e^-inf is 0
    tensor[:, data.VOCABULARY["<PAD>"], :] = float("-inf")
    tensor[:, data.VOCABULARY["<BOS>"], :] = float("-inf")
    return tensor

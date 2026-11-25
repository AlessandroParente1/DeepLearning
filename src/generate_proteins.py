import torch.nn.functional as F
import torch
from torch import nn

from protgen.data.make_data import AMINO_ACIDS, FAMILIES, MAX_DATA_LENGTH, VOCABULARY, VOCABULARY_SIZE
from protgen.models.mlp import CONTEXT_SIZE, MultiLayerPerceptron
from protgen.models.transformer import Transformer

# leave at 1 for neutral
TEMPERATURE = 0.5

# 0 is unguided
# 1 is entirely conditional (only the guided portion has any say)
# <0 and >1 extrapolates
#
# this modifies probability distribution differently from just changing temperature
# temperature makes likely things more or less likely (sharpen/flatten or w.e., basically it zooms in but doesn't change rank)
# example: if unguided says 10% 90% and guided says 15% 85% then scaling this up above 1 can get us to 60% 40% which temperature can never
W_CFG_LERP = 1

def main():
    model = Transformer().to("cuda")
    weights = torch.load("transformer_weights_1795377741.4822052.pt", weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    for family in FAMILIES:
        for _ in range(20):
            protein = generate_guided_protein(model, family, W_CFG_LERP)
            print(protein) # obviously dont do this manually forever

@torch.no_grad()
def generate_protein(model: nn.Module):
    context_tokens = [VOCABULARY["<PAD>"]] * CONTEXT_SIZE
    context_tokens[-1] = VOCABULARY["<BOS>"]

    index_to_token = {i: t for t, i in VOCABULARY.items()}

    protein = []
    prediction = None
    while prediction != VOCABULARY["<EOS>"]:

        # wrapper for gpu-ness
        # note the important [ ] to align shape with model expectations by adding one extra dim
        single_window = torch.tensor([context_tokens], device="cuda")
        input = F.one_hot(single_window, VOCABULARY_SIZE).float()
        # no need to re-shape because the window already has 1 x CONTEXT_SIZE dimensions

        logits = model(input)
        probabilities = logits.softmax(dim=-1)

        # note that these output an index, not a token, so we need to map
        next_token_index = torch.multinomial(probabilities, num_samples=1).item()
        
        #next_token_index = probs.argmax(dim=-1).item() # this one is fun to try for the famous sample-vs-max experiment
        # actually upon trying it, it never predicts EOS lol, we'd need to cut it short

        if next_token_index == VOCABULARY["<EOS>"]:
            break

        next_token = index_to_token[next_token_index]
        protein.append(next_token)
        context_tokens = context_tokens[1:] + [next_token_index] # very python-esque xdd should prolly just rotate or do a loop with modulo but w.e

    return "".join(protein)

@torch.no_grad()
def generate_conditional_protein(model: nn.Module, family: str):
    return generate_guided_protein(model, family, w=1)

@torch.no_grad()
def generate_unguided_protein(model: nn.Module, family: str):
    return generate_guided_protein(model, family, w=0)

@torch.no_grad()
def generate_guided_protein(model: nn.Module, family: str, w: float):

    index_to_token = {i: t for t, i in VOCABULARY.items()}

    # setting up two sequences, index 0 will be with family token, index 1 will be without
    tokens = torch.full((2, MAX_DATA_LENGTH), VOCABULARY["<PAD>"], device="cuda")
    tokens[0, 0] = VOCABULARY[family]
    tokens[1, 0] = VOCABULARY["<NO_FAMILY>"]
    tokens[:, 1] = VOCABULARY["<BOS>"]

    protein = []
    for position in range(1, MAX_DATA_LENGTH - 1):
        
        # both make their prediction
        conditional, unconditional = model(tokens)[:, position].log_softmax(dim=-1)
        
        # we do our mathy math math algebra
        # w from 0 to 1 lerps unguided to guided
        # less than 0 or more than 1 extrapolates in that direction
        logits = unconditional + w * (conditional - unconditional)
        probabilities = torch.softmax(logits / TEMPERATURE, dim=-1)

        next_token_index = torch.multinomial(probabilities, num_samples=1).item()
        if next_token_index == VOCABULARY["<EOS>"]:
            break

        protein.append(index_to_token[next_token_index])
        tokens[:, position + 1] = next_token_index

    return "".join(protein) 

# the transformer reads everything generated so far, not a fixed window, and takes token ids directly
# (it has its own embedding table) so there's no one_hot here
@torch.no_grad()
def generate_unformatted_input_protein(model: nn.Module):

    index_to_token = {i: t for t, i in VOCABULARY.items()}
    tokens = torch.full((1, MAX_DATA_LENGTH), VOCABULARY["<PAD>"], device="cuda")
    tokens[0, 0] = VOCABULARY["<BOS>"]

    protein = []
    for position in range(0, MAX_DATA_LENGTH - 1):
        logits = model(tokens)[0, position]
        probabilities = torch.softmax(logits / TEMPERATURE,dim=-1)

        next_token_index = torch.multinomial(probabilities, num_samples=1).item()
        if next_token_index == VOCABULARY["<EOS>"]:
            break

        protein.append(index_to_token[next_token_index])
        tokens[0, position + 1] = next_token_index

    return "".join(protein)


if __name__ == "__main__":
    main()
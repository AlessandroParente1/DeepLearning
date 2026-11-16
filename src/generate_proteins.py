import torch.nn.functional as F
import torch
from torch import nn

from protgen.data.make_data import AMINO_ACIDS, MAX_DATA_LENGTH, VOCABULARY, VOCABULARY_SIZE
from protgen.models.mlp import CONTEXT_SIZE, MultiLayerPerceptron
from protgen.models.transformer import Transformer


TEMPERATURE = 1


def main():
    model = Transformer().to("cuda")
    weights = torch.load("transformer_1dot5one_weights_1794778911.2550974.pt", weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    for _ in range(20):
        protein = generate_unformatted_input_protein(model)
        print(protein) # obviously dont do this manually forever


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

# yeah it turns out that models need sl
# the transformer reads everything generated so far, not a fixed window, and takes token ids directly
# (it has its own embedding table) so there's no one_hot here
@torch.no_grad()
def generate_unformatted_input_protein(model: nn.Module):
    index_to_token = {i: t for t, i in VOCABULARY.items()}

    # the model is never trained to predict these, but softmax still gives them a sliver of probability
    never_sampled = [VOCABULARY["<PAD>"], VOCABULARY["<BOS>"]]

    # the forward still asserts a full MAX_DATA_LENGTH input, so we keep a padded buffer and fill it in as we go
    # the causal mask means the padding after `position` can't affect the prediction at `position`
    tokens = torch.full((1, MAX_DATA_LENGTH), VOCABULARY["<PAD>"], device="cuda")
    tokens[0, 0] = VOCABULARY["<BOS>"]

    protein = []
    # position 0 is <BOS>, so there's room for MAX_DATA_LENGTH - 2 residues (the longest training protein)
    # multiplying by 2 rn while experimenting with some temperature that im okay going too far
    for position in range((MAX_DATA_LENGTH - 2) * 2):
        logits = model(tokens)[0, position] # the prediction for the token right after `position`
        logits[never_sampled] = float("-inf")
        probabilities = torch.softmax(logits / TEMPERATURE,dim=-1)


        next_token_index = torch.multinomial(probabilities, num_samples=1).item()
        if next_token_index == VOCABULARY["<EOS>"]:
            break

        protein.append(index_to_token[next_token_index])
        tokens[0, position + 1] = next_token_index

    return "".join(protein)


if __name__ == "__main__":
    main()
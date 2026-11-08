import torch.nn.functional as F
import torch
from torch import nn

from protgen.data.make_data import AMINO_ACIDS, VOCABULARY, VOCABULARY_SIZE
from protgen.models.mlp import CONTEXT_SIZE, MultiLayerPerceptron


def main():
    model = MultiLayerPerceptron().to("cuda")
    weights = torch.load("mlp1794165016.2621043.pt", weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    for _ in range(20):
        protein = generate_protein(model)
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


if __name__ == "__main__":
    main()
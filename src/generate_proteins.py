import torch.nn.functional as F
import torch
from torch import nn
import pandas as pd

from protgen.data.make_data import AMINO_ACIDS, MAX_DATA_LENGTH, VOCABULARY, VOCABULARY_SIZE
from protgen.models.mlp import CONTEXT_SIZE, MultiLayerPerceptron
from protgen.models.transformer import Transformer
from protgen.models.clasifier_reuse import Transformer_Classifier

# leave at 1 for neutral
TEMPERATURE = 1
TARGET_FAMILY = "HU"
GUIDANCE_STRENGTH = 1.0
#AMOUNT OF SEQUENCES TO GENERATE
NR_OF_SEQUENCES = 256


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Transformer().to(device)
    weights = torch.load("transformer_weights_for_classifier_1795441599.982232.pt", map_location = device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()

    #For clasifier:
    classifier = Transformer_Classifier().to(device)
    classifier_weights = torch.load(
        "classifier_weights_1795448369.030159.pt",
        map_location=device,
        weights_only=True
    )
    classifier.load_state_dict(classifier_weights)
    classifier.eval()


    #MAIN PROTEIN GENERATION LOOP CHANGE FUNCION FOR DIFFERENT GENRATIONS!
    #change the name of     output file with each run or it will be overwritten
    sequences = []

    for _ in range(NR_OF_SEQUENCES):
        protein = generate_classifier_guided_protein(model, classifier)
        sequences.append(protein)
        print(protein)

    df = pd.DataFrame({"sequence": sequences})
    df.to_csv(
        f"results/sequences/generated_proteins_with_classifier.csv",
        index=False
    )


#MLP BASED PROTEIN SEQ GENERATION
def generate_protein(model: nn.Module):
    context_tokens = [VOCABULARY["<PAD>"]] * CONTEXT_SIZE
    context_tokens[-1] = VOCABULARY["<BOS>"]

    index_to_token = {i: t for t, i in VOCABULARY.items()}

    protein = []
    prediction = None
    while prediction != VOCABULARY["<EOS>"]:

        # wrapper for gpu-ness
        # note the important [ ] to align shape with model expectations by adding one extra dim
        single_window = torch.tensor([context_tokens], device=device)
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


#TRANSFOMER BASED SEQ GENERATION 

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
    tokens = torch.full((1, MAX_DATA_LENGTH), VOCABULARY["<PAD>"], device=device)
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


#FOR CLASSIFIER GUIDED PROTEIN GENERATION:

def get_guided_probabilities(
    transformer_logits,
    classifier_probabilities,
    target_family,
    guidance_strength
):

    never_sampled = [
    VOCABULARY["<PAD>"],
    VOCABULARY["<BOS>"]
    ]

    target_index = ("AcyP", "HPr", "HU").index(target_family)

    transformer_log_probs = F.log_softmax(
        transformer_logits / TEMPERATURE,
        dim=-1
    )

    classifier_log_probs = torch.log(
        classifier_probabilities[:, target_index]
    )

    amino_acid_indices = torch.tensor(
        [VOCABULARY[aa] for aa in AMINO_ACIDS],
        device=transformer_logits.device
    )

    guided_scores = transformer_log_probs.clone()

    guided_scores[amino_acid_indices] += (
        guidance_strength * classifier_log_probs
    )

    guided_scores[never_sampled] = float("-inf")

    guided_probabilities = F.softmax(guided_scores, dim=-1)

    return guided_probabilities

@torch.no_grad()
def get_classifier_probabilities(classifier, tokens, position):
    candidates = tokens.repeat(len(AMINO_ACIDS), 1)

    amino_acid_indices = torch.tensor(
        [VOCABULARY[aa] for aa in AMINO_ACIDS],
        device=tokens.device
    )

    candidates[:, position + 1] = amino_acid_indices

    logits = classifier(candidates)

    class_logits = logits[:, position + 1, :]

    probabilities = torch.softmax(class_logits, dim=-1)

    return probabilities


@torch.no_grad()
def generate_classifier_guided_protein(model, classifier):

    index_to_token = {i: t for t, i in VOCABULARY.items()}

    never_sampled = [
        VOCABULARY["<PAD>"],
        VOCABULARY["<BOS>"]
    ]

    device = next(model.parameters()).device

    tokens = torch.full(
        (1, MAX_DATA_LENGTH),
        VOCABULARY["<PAD>"],
        device=device
    )
    tokens[0, 0] = VOCABULARY["<BOS>"]

    protein = []

    for position in range(MAX_DATA_LENGTH - 2):

        # Get transformer prediction for the next token
        transformer_logits = model(tokens)[0, position]

        # Prevent sampling PAD and BOS
        transformer_logits[never_sampled] = float("-inf")

        # Evaluate all 20 possible amino acids with the classifier
        classifier_probabilities = get_classifier_probabilities(
            classifier,
            tokens,
            position
        )

        #get probabilities after imput from classifier
        guided_probabilities = get_guided_probabilities(
            transformer_logits,
            classifier_probabilities,
            TARGET_FAMILY,
            GUIDANCE_STRENGTH
        )

        amino_acid_indices = [
            VOCABULARY[aa] for aa in AMINO_ACIDS
        ]

        #generate next amino acid
        next_token_index = torch.multinomial(
            guided_probabilities,
            num_samples=1
        ).item()

        if next_token_index == VOCABULARY["<EOS>"]:
            break

        protein.append(index_to_token[next_token_index])
        tokens[0, position + 1] = next_token_index

    return "".join(protein)



if __name__ == "__main__":
    main()
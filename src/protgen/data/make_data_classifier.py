#for data set:

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Constants for the dataset and model
FAMILIES = ("AcyP", "HPr", "HU")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
ALL_TOKENS = list(SPECIAL_TOKENS) + list(AMINO_ACIDS)


VOCABULARY = dict(zip(ALL_TOKENS, range(len(ALL_TOKENS))))
VOCABULARY_SIZE = len(VOCABULARY)
FAMILY_TO_INDEX = dict(zip(FAMILIES, range(len(FAMILIES))))

MAX_DATA_LENGTH = 127 + 2 #Longest sequence + <BOS> / <EOS>


class ProteinPrefixDataset:


    #building prefix samples:
    def __init__(self, path_to_data):

        #on deman implementation
        self.sequences = []
        self.protein_labels = []
        self.samples = []

        # Load raw CSV files
        self.df = pd.read_csv(path_to_data)

        #store one label per sequence independent on the number of prefixes we will generate
        self.sequences = self.df["sequence"].tolist()
        self.protein_labels = [FAMILY_TO_INDEX[label] for label in self.df["family"]]

        

        for protein_idx, sequence in enumerate(self.sequences):
            for prefix_length in range(1, len(sequence) + 1):
                self.samples.append((protein_idx, prefix_length))


        #generate everything implementation
        """
        #self.tokens = []
        #self.labels = []
        #self.lengths = []

        #encode prefixes:
        for sequence, label in self.df[["sequence", "family"]].values:
            for prefix_lenght in range (1, len(sequence) + 1):
                prefix = sequence[:prefix_lenght]

                #I need to encode the prefix using encode:
                tokens = self.encode(prefix)
                length = prefix_lenght

                #lets append the encoded prefix and protein labels
                self.tokens.append(tokens)
                
                self.labels.append(FAMILY_TO_INDEX[label])
                self.lengths.append(length)


        self.tokens = torch.stack(self.tokens)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)

        """

    # Convert prefixes to tokens:
    def encode(self, sequence):
            """Convert a string of residues to a tensor of token ids."""
            tokens = [VOCABULARY[residue] for residue in sequence]
            # Integer ids: nn.Embedding and F.cross_entropy both refuse a float tensor. <PAD> is id 0,
            # so the zeros are already the padding and only <BOS>/<EOS>/the residues are written in.
            x = torch.zeros(MAX_DATA_LENGTH, dtype=torch.long)
            # Add <BOS> and <EOS> tokens
            x[0] = VOCABULARY["<BOS>"]
            x[1:len(tokens)+1] = torch.tensor(tokens, dtype=torch.long)
            x[len(tokens)+1] = VOCABULARY["<EOS>"]
            return x

    """
    #so the dataloader could retrieve samples - prefixes and their labels:
    def __getitem__(self,index):
        return(
            self.tokens[index],
            self.labels[index],
            self.lengths[index]
        )
    """

    def __getitem__(self, index):

        #samples[index] - tells witch protein seq and prefix elnght we need
        protein_idx, prefix_length = self.samples[index]

        #retrieve the original protein seq
        sequence = self.sequences[protein_idx]

        #extract the prefix of interest
        prefix = sequence[:prefix_length]

        #create the token tensor on demand
        tokens = self.encode(prefix)
        #retrieve label
        label = self.protein_labels[protein_idx]
        #retrieve prefix lenght
        length = prefix_length

        return tokens, torch.tensor(label, dtype=torch.long), torch.tensor(length, dtype=torch.long)
         

    #Returns number of prefix samples:
    def __len__(self):

        #on demand:
        return len(self.samples)
        #return len(self.tokens)

    #allows to inspect the dataset

    def __repr__(self):
        return (
            f"ProteinPrefixDataset("
            f"number of samples = {len(self)}, "
            f"vocabulary size = {VOCABULARY_SIZE}, "
            f"number of classes = {len(FAMILY_TO_INDEX)}, "
            f"max data length = {MAX_DATA_LENGTH})"
        )

    """
    def __repr__(self):
        return f"ProteinPrefixDataset(number of samples = {len(self)}, vocabulary size = {VOCABULARY_SIZE}, number of classes = {len(FAMILY_TO_INDEX)}, max data length (hardcoded) = {MAX_DATA_LENGTH})"
    """
    #for inspecting prefixes:
    def decode(self, tokens):
        """Convert a tensor of token ids to a string of residues.

        Only the 20 residues have a character: <BOS>, <EOS>, <PAD> and the IGNORE_INDEX of a
        shifted target are dropped, so decoding either x or x_shifted gives the protein back.
        """
        idx_to_vocab = {index: token for token, index in VOCABULARY.items()}
        residues = set(AMINO_ACIDS)
        return "".join(idx_to_vocab[int(token)] for token in tokens
                        if idx_to_vocab.get(int(token)) in residues)


#for testing generating everything:
"""
dataset = ProteinPrefixDataset("data/train.csv")

print(dataset)
print("Number of samples:", len(dataset))

for i in range(5):
    tokens, label, length = dataset[i]

    print(f"\nSample {i}")
    print("Label:", label.item())
    print("Length:", length.item())
    print("Decoded sequence:", dataset.decode(tokens))

tokens, label, length = dataset[2]

print("Token IDs:", tokens)
print("First few tokens:", tokens[:8])
print("Length:", length)
print("EOS position:", length.item() + 1)

print("Tokens shape:", dataset.tokens.shape)
print("Labels shape:", dataset.labels.shape)
print("Lengths shape:", dataset.lengths.shape)

for i in range(5):
    tokens, label, length = dataset[i]

    eos_position = length.item() + 1

    assert tokens[eos_position].item() == VOCABULARY["<EOS>"]


Results:

ProteinPrefixDataset(number of samples = 1521889, vocabulary size = 23, number of classes = 3, max data length (hardcoded) = 129)
Number of samples: 1521889

Sample 0
Label: 1
Length: 1
Decoded sequence: M

Sample 1
Label: 1
Length: 2
Decoded sequence: MI

Sample 2
Label: 1
Length: 3
Decoded sequence: MIK

Sample 3
Label: 1
Length: 4
Decoded sequence: MIKK

Sample 4
Label: 1
Length: 5
Decoded sequence: MIKKQ
Token IDs: tensor([ 1, 13, 10, 11,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0])
First few tokens: tensor([ 1, 13, 10, 11,  2,  0,  0,  0])
Length: tensor(3)
EOS position: 4
Tokens shape: torch.Size([1521889, 129])
Labels shape: torch.Size([1521889])
Lengths shape: torch.Size([1521889])

"""
"""
# For testing on demand impementation:

dataset = ProteinPrefixDataset("data/train.csv")

print(dataset)
print("Number of samples:", len(dataset))

tokens, label, length = dataset[2]

print("Label:", label)
print("Length:", length)
print("Tokens:", tokens[:8])
print("Decoded sequence:", dataset.decode(tokens))

#Resukts:

ProteinPrefixDataset(number of samples = 1521889, vocabulary size = 23, number of classes = 3, max data length = 129)
Number of samples: 1521889
Label: tensor(1)
Length: tensor(3)
Tokens: tensor([ 1, 13, 10, 11,  2,  0,  0,  0])
Decoded sequence: MIK

"""
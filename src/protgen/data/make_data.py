import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Constants for the dataset and model
FAMILIES = ("AcyP", "HPr", "HU")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
ALL_TOKENS = list(SPECIAL_TOKENS) + list(AMINO_ACIDS)
IGNORE_INDEX = -1  # what F.cross_entropy(ignore_index=...) must be given for the generative task

VOCABULARY = dict(zip(ALL_TOKENS, range(len(ALL_TOKENS))))
VOCABULARY_SIZE = len(VOCABULARY)
FAMILY_TO_INDEX = dict(zip(FAMILIES, range(len(FAMILIES))))

MAX_DATA_LENGTH = 127 + 2 #Longest sequence + <BOS> / <EOS>

class ProteinDataset(Dataset):
    """
    A dataset class for protein sequences and their corresponding families.
    ----------------------------------------------------------------------
    Attributes (self.<attribute>):
    vocab: A dictionary mapping residues and special tokens to unique integer ids.
    fam_to_idx: A dictionary mapping family names to unique integer ids.
    df : A pandas DataFrame containing the raw CSV data.
    tokens: A torch tensor of token ids (N x max_length).
    labels: A torch tensor of integer ids (N).
    tokens_shifted: A torch tensor of token ids shifted + 1 (N x max_length).
    -----------------------------------------------------------------------
    x, x_shifted, y, length = dataset[index]
    - x: A tensor of token ids representing the protein sequence at the given index.
    - x_shifted: A tensor of token ids shifted to the left by one for the generative task.
    - y: An integer id representing the protein family at the given index.
    - length: The length of the protein sequence at the given index.
    -----------------------------------------------------------------------
    Other methods:
    - encode: Convert a string of residues to a tensor of token ids.
    - decode: Convert a tensor of token ids to a string of residues.
    """

    def __init__(self, path_to_data):
        # Load raw CSV files
        self.df = pd.read_csv(path_to_data)

        self.tokens, self.tokens_shifted, self.labels, self.lengths = [], [], [], []
        # encode proteins and familiy
        for sequence, label in self.df[["sequence", "family"]].values:
            tokens = self.encode(sequence)
            length = len(sequence)
            shifted_tokens = self.shift_tokens(tokens, length)
            # Append to lists
            self.tokens.append(tokens)
            self.tokens_shifted.append(shifted_tokens)
            self.labels.append(FAMILY_TO_INDEX[label])
            self.lengths.append(length)

        self.tokens = torch.stack(self.tokens)
        self.tokens_shifted = torch.stack(self.tokens_shifted)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

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

    def decode(self, tokens):
        """Convert a tensor of token ids to a string of residues.

        Only the 20 residues have a character: <BOS>, <EOS>, <PAD> and the IGNORE_INDEX of a
        shifted target are dropped, so decoding either x or x_shifted gives the protein back.
        """
        idx_to_vocab = {index: token for token, index in VOCABULARY.items()}
        residues = set(AMINO_ACIDS)
        return "".join(idx_to_vocab[int(token)] for token in tokens
                       if idx_to_vocab.get(int(token)) in residues)

    def shift_tokens(self, tokens : torch.Tensor, length : int):
        """Shift tokens to the left by one for the generative task."""
        shifted_tokens = torch.ones_like(tokens) * IGNORE_INDEX
        shifted_tokens[:length + 1] = tokens[1:length + 2] 
        return shifted_tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        label = self.labels[index]
        length = self.lengths[index]
        x = self.tokens[index]
        x_shifted = self.tokens_shifted[index]
        return x, x_shifted, label, length

    def __repr__(self):
        # Yes this format is a bit unconventional, no one cares, just keep it like this
        return f"ProteinDataset(number of samples = {len(self)}, vocabulary size = {VOCABULARY_SIZE}, number of classes = {len(FAMILY_TO_INDEX)}, max data length (hardcoded) = {MAX_DATA_LENGTH})"
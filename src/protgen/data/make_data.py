import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Constants for the dataset and model
FAMILIES = ("AcyP", "HPr", "HU")
AA = "ACDEFGHIKLMNPQRSTVWY"
special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
total_tokens = list(special_tokens) + list(AA)
IGNORE_INDEX = -1  # what F.cross_entropy(ignore_index=...) must be given for the generative task

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
        # Set token and family mappings
        self.vocab = dict(zip(total_tokens, range(len(total_tokens))))
        self.fam_to_idx = dict(zip(FAMILIES, range(len(FAMILIES))))

        # Load raw CSV files
        self.df = pd.read_csv(path_to_data)
        self.max_length = 127 + 2    #Longest sequence + <BOS> / <EOS>

        self.tokens, self.tokens_shifted, self.labels, self.lengths = [], [], [], []
        # encode proteins and familiy
        for sequence, label in self.df[["sequence", "family"]].values:
            tokens = self.encode(sequence)
            length = len(sequence)
            shifted_tokens = self.shift_tokens(tokens, length)
            # Append to lists
            self.tokens.append(tokens)
            self.tokens_shifted.append(shifted_tokens)
            self.labels.append(self.fam_to_idx[label])
            self.lengths.append(length)

        self.tokens = torch.stack(self.tokens)
        self.tokens_shifted = torch.stack(self.tokens_shifted)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def encode(self, sequence):
        """Convert a string of residues to a tensor of token ids."""
        tokens = [self.vocab[residue] for residue in sequence]
        # Integer ids: nn.Embedding and F.cross_entropy both refuse a float tensor. <PAD> is id 0,
        # so the zeros are already the padding and only <BOS>/<EOS>/the residues are written in.
        x = torch.zeros(self.max_length, dtype=torch.long)
        # Add <BOS> and <EOS> tokens
        x[0] = self.vocab["<BOS>"]
        x[1:len(tokens)+1] = torch.tensor(tokens, dtype=torch.long)
        x[len(tokens)+1] = self.vocab["<EOS>"]
        return x

    def decode(self, tokens):
        """Convert a tensor of token ids to a string of residues.

        Only the 20 residues have a character: <BOS>, <EOS>, <PAD> and the IGNORE_INDEX of a
        shifted target are dropped, so decoding either x or x_shifted gives the protein back.
        """
        idx_to_vocab = {index: token for token, index in self.vocab.items()}
        residues = set(AA)
        return "".join(idx_to_vocab[int(token)] for token in tokens
                       if idx_to_vocab.get(int(token)) in residues)

    def shift_tokens(self, tokens : torch.Tensor, length : int):
        """Shift tokens to the left by one for the generative task."""
        shifted_tokens = torch.ones_like(tokens) * IGNORE_INDEX
        shifted_tokens[:length + 1] = tokens[1:length + 2] 
        return shifted_tokens

    @property
    def vocab_size(self):
        """Size of a model's embedding table and output layer."""
        return len(self.vocab)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        label = self.labels[index]
        length = self.lengths[index]
        x = self.tokens[index]
        x_shifted = self.tokens_shifted[index]
        return x, x_shifted, label, length

    def __repr__(self):
        return f"ProteinDataset(num_samples={len(self)}, vocab_size={self.vocab_size}, num_classes={len(self.fam_to_idx)}, max_length={self.max_length})"
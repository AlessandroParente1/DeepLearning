import torch
from torch import nn
from protgen import data

# just using values close to the defaults of https://docs.pytorch.org/docs/2.15/generated/torch.nn.Transformer.html
MODEL_DIMENSIONS = 512
ATTENTION_HEADS_PER_BLOCK = 8
NUMBER_OF_BLOCKS = 6
HIDDEN_LAYER_SIZE = 2048
DROPOUT = 0.3


class Transformer(nn.Module):
    # the defaults are the original big model, so old checkpoints still load; pass smaller values for a smaller model
    def __init__(self, d_model=MODEL_DIMENSIONS, heads=ATTENTION_HEADS_PER_BLOCK, blocks=NUMBER_OF_BLOCKS,
                 ffn=HIDDEN_LAYER_SIZE, dropout=DROPOUT, num_labels=0):
        super().__init__()
        self.config = dict(d_model=d_model, heads=heads, blocks=blocks, ffn=ffn, dropout=dropout, num_labels=num_labels)

        # num_labels > 0 makes the model conditional (classifier-free guidance): a learned embedding per family, plus one
        # extra "null" label (index num_labels) meaning "no family given". It is added to every position, like the position embedding.
        # num_labels = 0 is the ordinary unconditional model, and its checkpoints have no label_embedding weights.
        self.num_labels = num_labels
        if num_labels > 0:
            self.label_embedding = nn.Embedding(num_labels + 1, d_model)

        # both of these are summed at the start to represent a token
        # token_embedding is a trained lookup table indexed by token identity (e.g. value "K")
        # position_embedding is a trained lookup table indexed by token position (e.g. index 5)
        self.token_embedding = nn.Embedding(data.VOCABULARY_SIZE, d_model)
        self.position_embedding = nn.Embedding(data.MAX_DATA_LENGTH, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # .blocks is not some special pytorch field, it's a regular old custom field
        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, ffn, dropout) for _ in range(blocks)])

        # since we do pre-norm
        self.final_norm = nn.LayerNorm(d_model)

        self.to_logits = nn.Linear(d_model, data.VOCABULARY_SIZE)


    # here input_tokens is a BATCH_SIZE x SOME_LENGTH tensor of tokens
    # labels (conditional models only): BATCH_SIZE tensor of family ids, with self.num_labels meaning "no label"
    # leaving labels out on a conditional model is the same as giving the null label for every protein
    def forward(self, input_tokens, labels=None):
        return self.to_logits(self.features(input_tokens, labels))

    # the final hidden state at every position, BATCH_SIZE x SOME_LENGTH x MODEL_DIMENSIONS
    # because of the causal mask, features[:, i] only depends on tokens 0..i, so it is a summary of the prefix
    # this is what the family classification head reads (classifier guidance) while the weights stay frozen
    def features(self, input_tokens, labels=None):
        # should work just fine with other lengths (between 1 and MAX_DATA_LENGTH inclusive)
        input_length = input_tokens.shape[1]

        # this is an "array_range" not the word "arrange"...
        # 0..length exclusive
        positions = torch.arange(input_length, device=input_tokens.device)

        # we look up the MODEL_DIMENSION-sized embeddings and sum them, that's our input, plain and simple
        embedded_input = self.token_embedding(input_tokens) + self.position_embedding(positions)
        if self.num_labels > 0:
            if labels is None:
                labels = torch.full((input_tokens.shape[0],), self.num_labels, dtype=torch.long, device=input_tokens.device)
            embedded_input = embedded_input + self.label_embedding(labels)[:, None, :]  # same label vector at every position
        embedded_input = self.embedding_dropout(embedded_input)

        # mask values are 0 or -inf, so that we can add them and make softmax ignore us (remember e^-inf=0)
        mask = nn.Transformer.generate_square_subsequent_mask(input_length, device=input_tokens.device)
        sequence = embedded_input
        for block in self.blocks:
             # this is the way to call forward() in the blocks
             # note that the first dimension of the sequence is still BATCH_SIZE
            sequence = block(sequence, mask)

        return self.final_norm(sequence)



class TransformerBlock(nn.Module):
    def __init__(self, d_model=MODEL_DIMENSIONS, heads=ATTENTION_HEADS_PER_BLOCK, ffn=HIDDEN_LAYER_SIZE, dropout=DROPOUT):
        super().__init__()

        # all we need :)
        self.attention_norm = nn.LayerNorm(d_model) # note that pre-norm is different from the 2017 paper but kind of just better and what we use now
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True) # batch_first because our dimensions are BATCH_SIZE x TOKENS not the other way around
        self.attention_dropout = nn.Dropout(dropout)

        # okay i guess we need this too
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn), 
            nn.ReLU(),
            nn.Linear(ffn, d_model),
            nn.Dropout(dropout),
        )


    def forward(self, input, mask):
        # remember that these "layer norms" that "normalize" dont just normalize, they also scale and shift everything with learned weights/biases
        # i have no clue why this is needed when the next layer should have weights and stuff anyway?
        # maybe explained in this 2016 paper? https://arxiv.org/abs/1607.06450 -- 2017 cites it
        normalized_input = self.attention_norm(input)

        # the _ we are discarding here is where the weights would be returned if we didn't pass in "need_weights=False"
        # query, key, and value are all computed from the input, (with their learned weights of course), that's why we pass in normalized_input three times
        attention_result, _ = self.attention(normalized_input, normalized_input, normalized_input, attn_mask=mask, need_weights=False)
        input_with_attention_result = input + self.attention_dropout(attention_result)

        normalized_attended_input = self.feed_forward_norm(input_with_attention_result)
        reflection_result = self.feed_forward(normalized_attended_input)

        # notice that we don't normalize the actual stream, we just let it grow
        output = input_with_attention_result + reflection_result
        return output



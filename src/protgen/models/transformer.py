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
    def __init__(self):
        super().__init__()

        # both of these are summed at the start to represent a token
        # token_embedding is a trained lookup table indexed by token identity (e.g. value "K")
        # position_embedding is a trained lookup table indexed by token position (e.g. index 5)
        self.token_embedding = nn.Embedding(data.VOCABULARY_SIZE, MODEL_DIMENSIONS)
        self.position_embedding = nn.Embedding(data.MAX_DATA_LENGTH + 1, MODEL_DIMENSIONS) # + 1 since we're prepending up to one token without adjusting max length
        self.embedding_dropout = nn.Dropout(DROPOUT)

        # .blocks is not some special pytorch field, it's a regular old custom field
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUMBER_OF_BLOCKS)])

        # since we do pre-norm
        self.final_norm = nn.LayerNorm(MODEL_DIMENSIONS)

        self.to_logits = nn.Linear(MODEL_DIMENSIONS, data.VOCABULARY_SIZE)


    # here input_tokens is a BATCH_SIZE x SOME_LENGTH tensor of tokens
    def forward(self, input_tokens):
        # should work just fine with other lengths (between 1 and MAX_DATA_LENGTH inclusive)
        input_length = input_tokens.shape[1]

        # this is an "array_range" not the word "arrange"...
        # 0..length exclusive
        positions = torch.arange(input_length, device=input_tokens.device)

        # we look up the MODEL_DIMENSION-sized embeddings and sum them, that's our input, plain and simple
        embedded_input = self.token_embedding(input_tokens) + self.position_embedding(positions)
        embedded_input = self.embedding_dropout(embedded_input)

        # mask values are 0 or -inf, so that we can add them and make softmax ignore us (remember e^-inf=0)
        mask = nn.Transformer.generate_square_subsequent_mask(input_length, device=input_tokens.device)
        sequence = embedded_input
        for block in self.blocks:
             # this is the way to call forward() in the blocks
             # note that the first dimension of the sequence is still BATCH_SIZE
            sequence = block(sequence, mask)

        normalized_output = self.final_norm(sequence)
        logits = self.to_logits(normalized_output)
        return logits # dimensions are BATCH_SIZE x SOME_LENGTH x VOCABULARY_SIZE



class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # all we need :)
        self.attention_norm = nn.LayerNorm(MODEL_DIMENSIONS) # note that pre-norm is different from the 2017 paper but kind of just better and what we use now
        self.attention = nn.MultiheadAttention(MODEL_DIMENSIONS, ATTENTION_HEADS_PER_BLOCK, dropout=DROPOUT, batch_first=True) # batch_first because our dimensions are BATCH_SIZE x TOKENS not the other way around
        self.attention_dropout = nn.Dropout(DROPOUT)

        # okay i guess we need this too
        self.feed_forward_norm = nn.LayerNorm(MODEL_DIMENSIONS)
        self.feed_forward = nn.Sequential(
            nn.Linear(MODEL_DIMENSIONS, HIDDEN_LAYER_SIZE), 
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, MODEL_DIMENSIONS),
            nn.Dropout(DROPOUT),
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



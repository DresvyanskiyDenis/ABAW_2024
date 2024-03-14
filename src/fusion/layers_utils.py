from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class ScaledDotProductAttention_MultiHead(nn.Module):

    def __init__(self, masking_strategy:str='padding'):
        super(ScaledDotProductAttention_MultiHead, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.masking_strategy = masking_strategy # can be 'filtering' or 'padding'


    def forward(self, query, key, value, mask=None):
        # query, key, value shapes:
        # [batch_size, num_heads, seq_len_query, dim_query],
        # [batch_size, num_heads, seq_len_key, dim_key],
        # [batch_size, num_heads, seq_len_value, dim_value]
        emb_dim_query = query.shape[-1]
        emb_dim_key = key.shape[-1]

        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(emb_dim_key)

        # Apply masking if provided
        if mask is not None:
            if self.masking_strategy == 'padding':
                attention_weights = attention_weights.masked_fill(mask == 0, -100000)

        # Softmax
        attention_weights = self.softmax(attention_weights)
        if mask is not None and self.masking_strategy == 'padding':
            attention_weights = attention_weights.masked_fill(mask == 0, 0)

        # Modify value
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class PositionWiseFeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout:float=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm= nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # feed-forward network
        x = self.layer_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_2(x)

        return x


class Add_and_Norm(nn.Module):

    def __init__(self, input_dim, dropout:Optional[float]=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)


    def forward(self, x1, residual):
        x = x1
        # apply dropout of needed
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        # add and then norm
        x = x + residual
        x = self.layer_norm(x)

        return x



class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim_query, input_dim_keys, input_dim_values, num_heads, dropout: Optional[float] = 0.1,
                 masking_strategy: str = 'padding'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_query = input_dim_query // num_heads
        self.head_dim_keys = self.head_dim_query # keys and queries should have the same dimensions
        self.head_dim_values = input_dim_values // num_heads
        self.dropout = dropout
        self.masking_strategy = masking_strategy

        # Initialize linear transformations for query, keys, and values
        self.query_w = nn.Linear(input_dim_query, self.num_heads * self.head_dim_query, bias=False)
        self.keys_w = nn.Linear(input_dim_keys, self.num_heads * self.head_dim_keys, bias=False)
        self.values_w = nn.Linear(input_dim_values, self.num_heads * self.head_dim_values, bias=False)
        self.ff_layer_after_concat = nn.Linear(self.num_heads * self.head_dim_values, input_dim_values, bias=False)

        self.attention = ScaledDotProductAttention_MultiHead(masking_strategy=self.masking_strategy)

        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        # query, keys, values shapes: [batch_size, seq_len, input_dim]
        # mask shape [batch_size, queries_len]
        batch_size, len_query, len_keys, len_values = queries.size(0), queries.size(1), keys.size(1), values.size(1)

        # linear transformation before attention
        queries = self.query_w(queries).view(batch_size, len_query, self.num_heads, self.head_dim_query).transpose(1, 2)  # [batch_size, num_heads, seq_len, dim]
        keys = self.keys_w(keys).view(batch_size, len_keys, self.num_heads, self.head_dim_keys).transpose(1, 2)  # [batch_size, num_heads, seq_len, dim]
        values = self.values_w(values).view(batch_size, len_values, self.num_heads, self.head_dim_values).transpose(1, 2)  # [batch_size, num_heads, seq_len, dim]

        # transform mask to the shape of [batch_size, num_heads, seq_len, seq_len]
        if mask is not None:
            # mask can be passed as Tuple of Tensors, in this case, there are two masks - one for queries and one for keys
            if isinstance(mask, torch.Tensor):
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, seq_len, 1]
                mask = mask.expand(batch_size, self.num_heads, len_query, len_keys)  # [batch_size, num_heads, seq_len, seq_len]
            elif isinstance(mask, tuple):
                mask_query, mask_keys = mask
                mask_query = mask_query.unsqueeze(1).unsqueeze(-1) # [batch_size, 1, seq_len, 1]
                mask_query = mask_query.expand(batch_size, self.num_heads, len_query, len_keys)
                mask_keys = mask_keys.unsqueeze(1).unsqueeze(1)# [batch_size, 1, 1, seq_len]
                mask_keys = mask_keys.expand(batch_size, self.num_heads, len_query, len_keys)
                # combine masks
                mask = mask_query * mask_keys

        # Attention mechanismI
        values, attention_weights = self.attention(queries, keys, values, mask=mask)

        # Concatenation
        out = values.transpose(1, 2).contiguous().view(batch_size, len_values, self.num_heads * self.head_dim_values)  # [batch_size, seq_len, num_heads * dim = input_dim]

        # Linear layer
        out = self.ff_layer_after_concat(out)

        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe=pe.permute(1, 0, 2) # [seq_len, batch_size, embedding_dim] -> [batch_size, seq_len, embedding_dim]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)



class Transformer_layer(nn.Module):

    def __init__(self, input_dims: Tuple, num_heads, dropout: Optional[float] = 0.1, positional_encoding: bool = True,
                 masking_strategy: str = 'padding'):
        super(Transformer_layer, self).__init__()
        self.positional_encoding = positional_encoding
        self.query_dim, self.key_dim, self.value_dim = input_dims
        # to perform full transformer layer, the input dimensions of the query, key, and value should be the same
        if self.query_dim != self.key_dim or self.query_dim != self.value_dim:
            raise ValueError("The input dimensions of the query, key, and value should be the same")
        self.num_heads = num_heads
        self.dropout = dropout
        self.masking_strategy = masking_strategy

        # initialize layers
        self.cross_attention = MultiHeadAttention(self.query_dim, self.key_dim, self.value_dim, num_heads, dropout=dropout,
                                                  masking_strategy=self.masking_strategy)
        self.feed_forward = PositionWiseFeedForward(self.query_dim, self.query_dim, dropout=dropout)
        self.add_norm_after_cross_attention = Add_and_Norm(self.query_dim, dropout=dropout)
        self.add_norm_after_ff = Add_and_Norm(self.query_dim, dropout=dropout)

        # calculate positional encoding
        if self.positional_encoding:
            self.positional_encoding_query = PositionalEncoding(self.query_dim)
            self.positional_encoding_key = PositionalEncoding(self.key_dim)
            self.positional_encoding_value = PositionalEncoding(self.value_dim)

    def forward(self, query, key, value, mask=None):
        # key, value, and query shapes: [batch_size, seq_len, input_dim]
        # positional encoding
        if self.positional_encoding:
            query = self.positional_encoding_query(query)
            key = self.positional_encoding_key(key)
            value = self.positional_encoding_value(value)

        # cross-attention
        residual = query
        x = self.cross_attention(keys=key, values=value, queries=query, mask=mask)
        x = self.add_norm_after_cross_attention(x, residual)

        # feed forward
        residual = x
        x = self.feed_forward(x)
        x = self.add_norm_after_ff(x, residual)

        return x


# example usage
if __name__ == "__main__":
    # create a transformer layer
    transformer = Transformer_layer(input_dims=(256, 256, 256), num_heads=8, dropout=0.1, positional_encoding=True)
    # create some dummy data
    query = torch.randn(8, 10, 256)  # [batch_size, seq_len, input_dim]
    key = torch.randn(8, 10, 256)  # [batch_size, seq_len, input_dim]
    value = torch.randn(8, 10, 256)  # [batch_size, seq_len, input_dim]
    # forward pass
    output = transformer(query, key, value)
    print(output.shape)  # torch.Size([32, 10, 512])
    print(output)
    print("Done!")


from typing import Optional

import torch

from pytorch_utils.layers.attention_layers import Transformer_layer


class VisualFusionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes:Optional[int]=None, num_regression_neurons:Optional[int]=None):
        super(VisualFusionModel, self).__init__()
        self.input_size = input_size # WARNING: two inputs should have the same size
        self.num_classes = num_classes
        self.num_regression_neurons = num_regression_neurons
        self.cross_attention_1_2 = Transformer_layer(input_dim=input_size[-1], num_heads=16, positional_encoding=True)
        self.cross_attention_2_1 = Transformer_layer(input_dim=input_size[-1], num_heads=16, positional_encoding=True)
        self.embeddings = torch.nn.Linear(input_size[-1] * 2, 256)
        if num_classes is not None:
            self.classification_head = torch.nn.Linear(256, num_classes)
        if num_regression_neurons is not None:
            self.regression_head = torch.nn.Linear(256, num_regression_neurons)


    def forward(self, x1, x2):
        # x1 and x2 have the same shape: [batch_size, seq_len, num_features]
        # cross attention
        x1_2 = self.cross_attention_1_2(query=x1, key=x2, value=x2)
        x2_1 = self.cross_attention_2_1(query=x2, key=x1, value=x1)
        # concatenate
        x = torch.cat([x1_2, x2_1], dim=-1) # -> [batch_size, seq_len, num_features * 2]
        # embeddings
        x = self.embeddings(x)
        output = []
        if self.num_classes is not None:
            output.append(self.classification_head(x))
        if self.num_regression_neurons is not None:
            output.append(self.regression_head(x))
        return output
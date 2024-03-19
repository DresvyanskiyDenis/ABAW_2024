import sys

sys.path.append('src')

import numpy as np
import torch
import torch.nn as nn

from audio.models.attention_layers import *


class TestModelSTP(torch.nn.Module):
    def __init__(self, a_f_size=1024, v_f_size=256, av_f_size=256, num_regression_neurons=2):
        super(TestModelSTP, self).__init__()
        self.a_f_size = a_f_size
        self.v_f_size = v_f_size
        self.av_f_size = av_f_size

        self.num_regression_neurons = num_regression_neurons


        self.a_fc1 = torch.nn.Linear(self.a_f_size, self.av_f_size)
        self.a_relu = torch.nn.ReLU()
        
        self.attention_va_fusion = AttentionFusionModel(self.v_f_size, self.v_f_size, out_size=self.v_f_size)


        self.attention_fusion_model = AttentionFusionModel(self.av_f_size, self.av_f_size, out_size=self.av_f_size)
        
        self.regressor = torch.nn.Linear(self.av_f_size, self.num_regression_neurons)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x_a_va, x_v_va = x
        x_v_v = x_v_va[:, 0, :, :]
        x_v_a = x_v_va[:, 1, :, :]

        a_f = self.a_relu(self.a_fc1(x_a_va))

        v_f = self.attention_va_fusion(x_v_v, x_v_a)

        x = self.attention_fusion_model(a_f, v_f)

        x = self.regressor(x)
        x = self.tanh(x)
        return x


class TestModelMean(torch.nn.Module):
    def __init__(self, a_f_size=1024, v_f_size=256, av_f_size=256, num_regression_neurons=2):
        super(TestModelMean, self).__init__()
        self.a_f_size = a_f_size
        self.v_f_size = v_f_size
        self.av_f_size = av_f_size

        self.num_regression_neurons = num_regression_neurons


        self.a_fc1 = torch.nn.Linear(self.a_f_size, self.av_f_size)
        self.a_relu = torch.nn.ReLU()
        
        self.attention_va_fusion = AttentionFusionModel(self.v_f_size, self.v_f_size, out_size=self.v_f_size, aggregation='mean')


        self.attention_fusion_model = AttentionFusionModel(self.av_f_size, self.av_f_size, out_size=self.av_f_size, aggregation='mean')
        
        self.regressor = torch.nn.Linear(self.av_f_size, self.num_regression_neurons)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x_a_va, x_v_va = x
        x_v_v = x_v_va[:, 0, :, :]
        x_v_a = x_v_va[:, 1, :, :]

        a_f = self.a_relu(self.a_fc1(x_a_va))

        v_f = self.attention_va_fusion(x_v_v, x_v_a)

        x = self.attention_fusion_model(a_f, v_f)

        x = self.regressor(x)
        x = self.tanh(x)
        return x

class TestModelConcat(torch.nn.Module):
    def __init__(self, a_f_size=1024, v_f_size=256, av_f_size=256, num_regression_neurons=2):
        super(TestModelConcat, self).__init__()
        self.a_f_size = a_f_size
        self.v_f_size = v_f_size
        self.av_f_size = av_f_size

        self.num_regression_neurons = num_regression_neurons


        self.a_fc1 = torch.nn.Linear(self.a_f_size, self.av_f_size)
        self.a_relu = torch.nn.ReLU()
        
        self.attention_va_fusion = AttentionFusionModel(self.v_f_size, self.v_f_size, out_size=self.v_f_size, aggregation='concat')

        self.attention_fusion_model = AttentionFusionModel(self.av_f_size, self.av_f_size, out_size=self.av_f_size, aggregation='concat')
        
        self.regressor = torch.nn.Linear(self.av_f_size, self.num_regression_neurons)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x_a_va, x_v_va = x
        x_v_v = x_v_va[:, 0, :, :]
        x_v_a = x_v_va[:, 1, :, :]

        a_f = self.a_relu(self.a_fc1(x_a_va))

        v_f = self.attention_va_fusion(x_v_v, x_v_a)

        x = self.attention_fusion_model(a_f, v_f)

        x = self.regressor(x)
        x = self.tanh(x)
        return x


class StatPoolLayer(torch.nn.Module):
    def __init__(self, dim):
        super(StatPoolLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim)
        std = x.std(dim=self.dim)
        return torch.cat([mean, std], dim=self.dim)


class AttentionFusionModel(torch.nn.Module):
    def __init__(self, e1_num_features: int, e2_num_features: int, out_size: int, num_heads: int = 8, aggregation: str = 'stp'):
        # build cross-attention layers
        super(AttentionFusionModel, self).__init__()
        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.num_heads = num_heads
        self.out_size = out_size
        self.aggregation = aggregation
        self._build_cross_attention_modules(self.e1_num_features, self.e2_num_features, self.out_size)

    def _build_cross_attention_modules(self, e1_num_features: int, e2_num_features: int, out_size: int):
        self.e1_cross_att_layer_1 = TransformerLayer(input_dim=e1_num_features, num_heads=self.num_heads,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_1 = TransformerLayer(input_dim=e2_num_features, num_heads=self.num_heads,
                                                      dropout=0.1, positional_encoding=True)
        self.e1_cross_att_layer_2 = TransformerLayer(input_dim=e1_num_features, num_heads=self.num_heads,
                                                      dropout=0.1, positional_encoding=True)
        self.e2_cross_att_layer_2 = TransformerLayer(input_dim=e2_num_features, num_heads=self.num_heads,
                                                      dropout=0.1, positional_encoding=True)
        
        self.stp_pool = StatPoolLayer(dim=2)

        if self.aggregation == 'mean':
            self.cross_ett_dense_layer = torch.nn.Linear(e1_num_features, out_size)
        else:
            self.cross_ett_dense_layer = torch.nn.Linear(e1_num_features + e2_num_features, out_size)

        self.cross_att_batch_norm = torch.nn.BatchNorm1d(out_size)
        self.cross_att_activation = torch.nn.ReLU()

    def forward_cross_attention(self, e1, e2):
        # cross attention 1
        e1 = self.e1_cross_att_layer_1(key=e1, value=e1,
                                       query=e2)  # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer_1(key=e2, value=e2,
                                       query=e1)  # Output shape (batch_size, sequence_length, e2_num_features)
        # cross attention 2
        e1 = self.e1_cross_att_layer_2(key=e1, value=e1,
                                       query=e2)  # Output shape (batch_size, sequence_length, e1_num_features)
        e2 = self.e2_cross_att_layer_2(key=e2, value=e2,
                                       query=e1)  # Output shape (batch_size, sequence_length, e2_num_features)
        # concat e1 and e2
        if self.aggregation == 'stp':
            x = torch.stack((e1, e2),
                            dim=2)  # Output shape (batch_size, sequence_length, 2, 256)
            x = self.stp_pool(x)
        elif self.aggregation == 'mean':
            x = torch.stack((e1, e2),
                            dim=2)  # Output shape (batch_size, sequence_length, 2, 256)
            x = x.mean(dim=2)
        else:
            x = torch.concat((e1, e2), dim=-1)

        x = self.cross_ett_dense_layer(x)
        x = x.permute(0, 2, 1)
        x = self.cross_att_batch_norm(x)
        x = x.permute(0, 2, 1)
        x = self.cross_att_activation(x)
        return x

    def forward(self, feature_set_1, feature_set_2):
        # features is a list of tensors
        # every element of the list has a shape of (batch_size, sequence_length, num_features)
        e1, e2 = feature_set_1, feature_set_2
        # cross attention
        x = self.forward_cross_attention(e1, e2)
        return x

if __name__ == "__main__":      
    inp_a = torch.zeros((1, 20, 1024))
    inp_v = torch.zeros((1, 2, 20, 256))
    for m in [TestModelConcat, TestModelMean, TestModelSTP]:
        model = m(a_f_size=1024, v_f_size=256, av_f_size=256, num_regression_neurons=2)

        res = model((inp_a, inp_v))
        print(res)
        print(res.shape)
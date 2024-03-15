import sys

sys.path.append('src')

import numpy as np
import torch
import torch.nn as nn

from audio.models.attention_layers import *

class TestModel(torch.nn.Module):
    def init(self, a_f_size=1024, v_f_size=256, av_f_size=256, num_classes=8):
        super(TestModel, self).init()
        self.a_f_size = a_f_size
        self.v_f_size = v_f_size
        self.av_f_size = av_f_size

        self.out_size = num_classes

        self.a_downsample = torch.nn.Linear(self.a_f_size, self.av_f_size)

        self.v_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.v_f_size, self.v_f_size, kernel_size=5, stride=3, dilation=2),
            torch.nn.BatchNorm1d(self.v_f_size),
            torch.nn.AdaptiveAvgPool1d(20),
            torch.nn.ReLU(),

            torch.nn.Conv1d(self.v_f_size, self.v_f_size, kernel_size=5, stride=2),
            torch.nn.BatchNorm1d(self.v_f_size),
            torch.nn.AdaptiveAvgPool1d(4),
            torch.nn.ReLU(),
        )
        
        self.attention_fusion_model = AttentionFusionModel(self.av_f_size, self.av_f_size, out_size=self.av_f_size)
        
        self.classifier = torch.nn.Linear(self.av_f_size, self.out_size)

    def forward(self, x):
        x_a, x_v = x

        a_f = self.a_downsample(x_a)

        x_v = x_v.permute(0, 2, 1)
        v_f = self.v_downsample(x_v)
        v_f = v_f.permute(0, 2, 1)

        x = self.attention_fusion_model(a_f, v_f)
        x = self.classifier(x)
        x = torch.concat([x[:,i,:].unsqueeze(1).repeat(1, 5, 1) for i in range(4)], dim=1)
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
    def __init__(self, e1_num_features: int, e2_num_features: int, out_size: int, num_heads: int = 8):
        # build cross-attention layers
        super(AttentionFusionModel, self).__init__()
        self.e1_num_features = e1_num_features
        self.e2_num_features = e2_num_features
        self.num_heads = num_heads
        self.out_size = out_size
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
        x = torch.stack((e1, e2),
                      dim=2)  # Output shape (batch_size, sequence_length, 2, 256)
        x = self.stp_pool(x)

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
    sampling_rate = 16000
    inp_a = torch.zeros((4, 4, 1024))
    inp_v = torch.zeros((4, 20, 256))
    model = TestModel(a_f_size=1024, v_f_size=256, av_f_size=256, out_size=8)

    res = model((inp_a, inp_v))
    print(res)
    print(res.shape)
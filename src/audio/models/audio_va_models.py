import sys

sys.path.append('src/audio')

import numpy as np
import torch
import torch.nn as nn

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from models.attention_layers import TransformerLayer


class VAModelV1(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.gru = nn.GRU(input_size=1024, hidden_size=256, dropout=.5, num_layers=2, batch_first=True)
        self.relu = nn.Tanh()

        self.f_size = 256
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=5, stride=3, dilation=2),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(20),
            torch.nn.ReLU(),
        )
        
        self.feature_downsample = nn.Linear(self.f_size, 2)
        self.tanh_va = nn.Tanh()
        
        self.init_weights()
        self.unfreeze_last_n_blocks(2)
 
    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        # freeze all wav2vec
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
 
        # unfreeze last n transformer blocks
        for i in range(0, num_blocks):
            for param in self.wav2vec2.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, x):
        outputs = self.wav2vec2(x)
        x, h = self.gru(outputs[0])

        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)

        x = x.permute(0, 2, 1)
        x = self.feature_downsample(x)
        x = self.tanh_va(x)
        return x


class VAModelV2(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        
        self.tl1 = TransformerLayer(input_dim=1024, num_heads=32, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=1024, num_heads=16, dropout=0.1, positional_encoding=True)
        
        self.f_size = 1024
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=5, stride=3, dilation=2),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(20),
            torch.nn.ReLU(),
        )
        
        self.feature_downsample = nn.Linear(self.f_size, 2)
        self.tanh_va = nn.Tanh()

        self.init_weights()
        self.unfreeze_last_n_blocks(2)
        
    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        # freeze all wav2vec
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # unfreeze last n transformer blocks
        for i in range(0, num_blocks):
            for param in self.wav2vec2.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.wav2vec2(x)[0]

        x = self.tl1(query=x, key=x, value=x)
        x = self.tl2(query=x, key=x, value=x)

        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)

        x = x.permute(0, 2, 1)
        x = self.feature_downsample(x)
        x = self.tanh_va(x)
        return x
    
    
class VAModelV3(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        
        self.tl1 = TransformerLayer(input_dim=1024, num_heads=32, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=1024, num_heads=16, dropout=0.1, positional_encoding=True)
        
        self.f_size = 1024
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=5, stride=3, dilation=2),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(20),
            torch.nn.ReLU(),
        )
        
        self.feature_downsample = nn.Linear(self.f_size, 2)
        self.tanh_va = nn.Tanh()
        
        self.init_weights()
        self.unfreeze_last_n_blocks(4)
        
    def freeze_conv_only(self):
        # freeze conv
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False
            
    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        # freeze all wav2vec
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # unfreeze last n transformer blocks
        for i in range(0, num_blocks):
            for param in self.wav2vec2.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.wav2vec2(x)[0]

        x = self.tl1(query=x, key=x, value=x)
        x = self.tl2(query=x, key=x, value=x)

        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)

        x = x.permute(0, 2, 1)
        x = self.feature_downsample(x)
        x = self.tanh_va(x)
        return x


if __name__ == "__main__":      
    sampling_rate = 16000
    inp_v = torch.zeros((4, sampling_rate * 4))
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    model_cls = VAModelV3

    model = model_cls.from_pretrained(model_name)

    res = model(inp_v)
    print(res)
    print(res.shape)
from typing import Tuple

import torch

from pytorch_utils.layers.attention_layers import Transformer_layer


class UniModalTemporalModel(torch.nn.Module):
    def __init__(self, input_shape:Tuple[int, int], num_classes:int, num_regression_neurons:int):
        super(UniModalTemporalModel, self).__init__()
        self.num_timesteps, self.num_features = input_shape # (num_time_steps, num_features)
        self.num_classes = num_classes
        self.num_regression_neurons = num_regression_neurons
        self.__initialize_temporal_part()

        self.classifier = torch.nn.Linear(in_features=self.num_features//4, out_features=self.num_classes)
        self.regressor = torch.nn.Linear(in_features=self.num_features//4, out_features=self.num_regression_neurons)


    def __initialize_temporal_part(self):
        # make the first part of the model as torch list
        self.first_temporal_part = torch.nn.ModuleList()
        self.second_temporal_part = torch.nn.ModuleList()
        self.third_temporal_part = torch.nn.ModuleList()
        # first part of the model
        self.first_temporal_part.append(Transformer_layer(input_dim=self.num_features, num_heads=8, dropout=0.1, positional_encoding=True))
        self.first_temporal_part.append(Transformer_layer(input_dim=self.num_features, num_heads=8, dropout=0.1, positional_encoding=True))
        # reduction of feature dimension by 2
        self.feature_reduction_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features //2, kernel_size=1, stride=1),
            torch.nn.BatchNorm1d(num_features=self.num_features // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
        )
        # two more transformer layers
        self.second_temporal_part.append(Transformer_layer(input_dim=self.num_features//2, num_heads=8, dropout=0.1, positional_encoding=True))
        self.second_temporal_part.append(Transformer_layer(input_dim=self.num_features//2, num_heads=8, dropout=0.1, positional_encoding=True))

        self.feature_reduction_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.num_features//2, out_channels=self.num_features // 4, kernel_size=1, stride=1),
            torch.nn.BatchNorm1d(num_features=self.num_features // 4),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
        )
        # two more transformer layers
        self.third_temporal_part.append(Transformer_layer(input_dim=self.num_features//4, num_heads=4, dropout=0.1, positional_encoding=True))
        self.third_temporal_part.append(Transformer_layer(input_dim=self.num_features//4, num_heads=4, dropout=0.1, positional_encoding=True))


    def forward(self, x):
        # first temporal part
        for layer in self.first_temporal_part:
            x = layer(x, x, x)
        # reduction
        x = x.permute(0, 2, 1)
        x = self.feature_reduction_1(x)
        x = x.permute(0, 2, 1)
        # second temporal part
        for layer in self.second_temporal_part:
            x = layer(x, x, x)
        # reduction
        x = x.permute(0, 2, 1)
        x = self.feature_reduction_2(x)
        x = x.permute(0, 2, 1)
        # third temporal part
        for layer in self.third_temporal_part:
            x = layer(x, x, x)
        # classification
        class_output = self.classifier(x)
        regression_output = self.regressor(x)
        return class_output, regression_output



if __name__ == "__main__":
    model = UniModalTemporalModel(input_shape=(20,256), num_classes=8, num_regression_neurons=2)
    print(model)
    x = torch.rand(32, 20, 256)
    print(model(x))
    import torchsummary
    torchsummary.summary(model, (20, 256), device='cpu')

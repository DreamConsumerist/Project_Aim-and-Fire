## Regression targets could be another output of this model to smooth aiming.
## Transition classes could achieve earlier response rate as the model detects the transition into a stable state.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Only apply 1x1 convolution if input and output channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        out = F.pad(x, (pad, 0))  # pad only on the left

        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = F.pad(out, (pad, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCNGestureClassifier(nn.Module):
    def __init__(self, num_features=63, num_classes=5, num_blocks=4, hidden_channels=64, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        dilations = [2**i for i in range(num_blocks)]
        in_ch = num_features
        for d in dilations:
            layers.append(ResidualBlock(in_ch, hidden_channels, kernel_size, dilation=d, dropout=dropout))
            in_ch = hidden_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, features] -> transpose to [batch, features, seq_len]
        x = x.transpose(1,2)
        x = self.network(x)
        # take last time step
        x = x[:,:,-1]
        return self.fc(x)

model = TCNGestureClassifier()
x = torch.randn(2, 30, 63)  # batch=2, seq_len=30, features=63
out = model(x)
print(out.shape)  # should be [2, 5] for num_classes=5
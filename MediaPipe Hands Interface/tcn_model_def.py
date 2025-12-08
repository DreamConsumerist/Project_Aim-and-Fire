import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Only apply 1x1 convolution if input/output channels differ
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCNGestureClassifier(nn.Module):
    def __init__(self, num_features=63, num_classes=3, num_blocks=3, hidden_channels=64, kernel_size=3, dropout=0.2):
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
        x = x.transpose(1,2)  # [batch, features, seq_len]
        x = self.network(x)
        x = x[:,:,-1]          # take last time step
        return self.fc(x)
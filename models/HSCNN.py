import torch
import torch.nn as nn
import torch.nn.functional as F
from models.KAN_head import KAN


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y


class HSCNN(nn.Module):
    def __init__(self, config):
        super(HSCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            config.SOLVER["pca_components"], 32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

        self.se_block = SEBlock(64)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            config.MODEL["patch_size"] * config.MODEL["patch_size"] * 64, 128
        )
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, config.MODEL["num_classes"])
        self.dropout2 = nn.Dropout(0.4)
        # self.kan_head = KAN(config.MODEL["kan_list"])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x = x1 + x2

        x = self.se_block(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.kan_head(x)
        x = self.dropout2(x)

        return x

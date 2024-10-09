import torch.nn.functional as F
from torch import nn


class baseCNN(nn.Module):
    def __init__(self):
        super(baseCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 18)
        # self.fc3 = nn.Linear(128, 18)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1, 3, 3)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        return x

import torch.nn as nn
import torch.nn.functional as F

from models.KAN_head import KAN

class CNN_plus_KAN(nn.Module):
    def __init__(self, in_channels , C1_num, num_classes, kan_list, dropout):
        super(CNN_plus_KAN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=C1_num, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=C1_num, out_channels=3*C1_num, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.KAN_head = KAN(kan_list)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.KAN_head(x)
        x = self.dropout2(x)
        return F.softmax(x, dim=1)
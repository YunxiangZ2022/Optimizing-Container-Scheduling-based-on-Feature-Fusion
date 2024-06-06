import torch
import torch.nn as nn

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 2), padding = (0, 1))
        self.bn0 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 2), stride=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 2), stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2), stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv55 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 2), stride=1)
        self.bn55 = nn.BatchNorm2d(128)
        self.conv66 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 2), stride=1)
        self.bn66 = nn.BatchNorm2d(256)
        self.fc0 = nn.Linear(1792, 768)
        self.bn5 = nn.BatchNorm1d(768)
        self.fc1 = nn.Linear(768, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn7 = nn.BatchNorm1d(64)
        # 连接后
        self.fc3 = nn.Linear(64, 48)
        self.fc4 = nn.Linear(48, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        x = self.bn0(self.relu(self.conv0(x)))
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn55(self.relu(self.conv55(x)))
        x = self.bn66(self.relu(self.conv66(x)))
        # flatten
        x = x.view(batch_size, -1)
        # print('in_features:', x.size(-1))
        x = self.bn5(self.relu(self.fc0(x)))
        x = self.bn6(self.relu(self.fc1(x)))
        x = self.bn7(self.relu(self.fc2(x)))

        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        return x
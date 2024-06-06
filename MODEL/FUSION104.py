import torch
import torch.nn as nn

stack = 10
tier = 4

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 2), padding = (0, 1))
        self.bn0 = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 2), stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 2), stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2), stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 2), stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc0 = nn.Linear(1280, 768)
        self.bn5 = nn.BatchNorm1d(768)
        self.fc1 = nn.Linear(768, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.bn7 = nn.BatchNorm1d(32)
        
        self.lstm = nn.LSTM(input_size=tier, hidden_size=64, num_layers=4, batch_first=True, bidirectional=True)
        self.fc3 = nn.Linear(128, 32)

        self.fc4 = nn.Linear(5 * stack, 32)
        
        # 连接后
        self.fc5 = nn.Linear(96, 48)
        self.fc6 = nn.Linear(48, 32)
        self.fc7 = nn.Linear(32, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        z = x.view(batch_size, stack, tier)
        x = self.bn0(self.relu(self.conv0(x)))
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        # flatten
        x = x.view(batch_size, -1)
        # print('in_features:', x.size(-1))
        x = self.bn5(self.relu(self.fc0(x)))
        x = self.bn6(self.relu(self.fc1(x)))
        x = self.bn7(self.relu(self.fc2(x)))
        
        z, _ = self.lstm(z, None)
        z = self.relu(self.fc3(z[:, -1, :]))

        y = y.view(batch_size, -1)
        y = self.relu(self.fc4(y))

        xyz = torch.cat((x, y, z), dim=1)
        xyz = self.relu(self.fc5(xyz))
        xyz = self.relu(self.fc6(xyz))
        xyz = self.relu(self.fc7(xyz))
        return xyz
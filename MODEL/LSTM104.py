import torch
import torch.nn as nn

stack = 10
tier = 4

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=tier, hidden_size=64, num_layers=4, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 64)
        # 连接后
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        z = x.view(batch_size, stack, tier)
        z, _ = self.lstm(z, None)
        z = self.relu(self.fc1(z[:, -1, :]))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.relu(self.fc4(z))
        return z
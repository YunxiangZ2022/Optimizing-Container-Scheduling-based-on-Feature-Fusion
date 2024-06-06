import torch
import torch.nn as nn

stack = 10

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5 * stack, 64)
        # 连接后
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, y):
        batch_size = x.size(0)
        y = y.view(batch_size, -1)
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc3(y))
        y = self.relu(self.fc4(y))
        return y
import torch
from torch import nn

class DQNetwork(nn.Model):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.nn.functional.leaky_relu(self.fc1(state), negative_slope=0.01) # how much is it tilted
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        return self.fc3(x)
    

import torch
from torch import nn 

class NeuralNN(nn.Module):
    def __init__(self):
        super(NeuralNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1) # Minor note: softmax(dim=0) should be softmax(dim=1) for batch-wise softmax during training/export. - from Ai
                                              # This makes it work per batch of inputs. STM32Cube.AI expects batch-wise softmax (dim=1), not over features.
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def hard_sigmoid(x, temperature=0.1):
    return torch.sigmoid(x / temperature)


class ToyModel(nn.Module):
    def __init__(self, num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor):
        super().__init__()
        self.fc1 = nn.Linear(num_nodes, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, num_nodes)

    def forward(self, x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes):
        x = F.relu(self.fc1(b10.float()))        
        # x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return hard_sigmoid(logits), hard_sigmoid(logits)
    
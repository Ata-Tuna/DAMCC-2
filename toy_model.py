import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Toy model with additional layers
class ToyModel(nn.Module):
    def __init__(self, num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor):
        super().__init__()
        self.fc1 = nn.Linear(num_nodes, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, num_nodes)

    def forward(self, x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes):
        print(b10)
        sys.exit(0)
        x = F.relu(self.fc1(b10))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.sigmoid(logits)
    
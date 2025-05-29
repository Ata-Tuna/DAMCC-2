import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def hard_sigmoid(x, temperature=0.1):
    return torch.sigmoid(x / temperature)

hidden_dim = 8  # Define a hidden dimension for the GCN layers

class ToyModel(nn.Module):
    def __init__(self, num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GCNConv(num_nodes, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_nodes * num_nodes)
        
    def forward(self, x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes):
        # Use b10 as input features and a1 as edge_index
        edge_index = a1.nonzero(as_tuple=False).t()  # Convert adjacency matrix to edge_index format
        x = F.relu(self.gcn1(b1.float(), edge_index))
        x = F.relu(self.gcn2(x, edge_index))
    
        # Generate logits for adjacency matrix
        x = self.fc(x)
        logits = x.view(-1, self.num_nodes, self.num_nodes)
        
        # Apply hard_sigmoid to generate incidence matrix
        incidence_matrix = hard_sigmoid(logits)
        return incidence_matrix, incidence_matrix

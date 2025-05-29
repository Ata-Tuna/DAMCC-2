import torch
from cc_embedder import CCEmbedder
from model import AutoregressiveSubsetSampler  # Import from model.py
from loss_functions import row_wise_permutation_invariant_loss  # Import the loss function
import sys
sys.path.append('..')  # Adjust the path as necessary to import from the parent directory
from utils.utils import hard_sigmoid  # Import utility functions


class Damcc(torch.nn.Module):
    def __init__(self, num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor):
        super().__init__()

        self.cc_embedder = CCEmbedder(feature_n_0_cells, feature_n_1_cells, feature_n_2_cells)

        self.decoder_0_cells = AutoregressiveSubsetSampler(
            num_nodes, n_features, size_g, n_features, new_cell_factor, min_nonzero=2, max_nonzero=2)
        self.decoder_1_cells = AutoregressiveSubsetSampler(
            num_nodes, n_features, size_g, n_features, new_cell_factor, min_nonzero=3, max_nonzero=5)
        self.linear_0_cells = torch.nn.Linear(feature_n_0_cells, num_nodes * num_nodes)
        self.linear_1_cells = torch.nn.Linear(feature_n_1_cells, num_nodes * num_nodes)

    def forward(self,             
            x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20,
            num_nodes):

        embeddings_1, embeddings_2 = self.cc_embedder(x_0, x_1, x_2, a1, a2, coa2, b1, b2)

        # Apply linear transformations
        output_1 = self.linear_0_cells(embeddings_1).view(-1, num_nodes, num_nodes)
        output_2 = self.linear_1_cells(embeddings_2).view(-1, num_nodes, num_nodes)

        return output_1, output_2

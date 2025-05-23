import torch
from cc_embedder import CCEmbedder
from model import AutoregressiveSubsetSampler  # Import from model.py
from loss_functions import row_wise_permutation_invariant_loss  # Import the loss function
import sys

class Damcc(torch.nn.Module):
    def __init__(self, num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor):
        super().__init__()

        self.cc_embedder = CCEmbedder(feature_n_0_cells, feature_n_1_cells, feature_n_2_cells)

        self.decoder_0_cells = AutoregressiveSubsetSampler(
            num_nodes, n_features, size_g, n_features, new_cell_factor, min_nonzero=2, max_nonzero=2)
        self.decoder_1_cells = AutoregressiveSubsetSampler(
            num_nodes, n_features, size_g, n_features, new_cell_factor, min_nonzero=3, max_nonzero=5)

    def forward(self,             
            x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20,
            num_nodes, 
            get_ll=False, delta_edges=None):

        embeddings_1, embeddings_2 = self.cc_embedder(x_0, x_1, x_2, a1, a2, coa2, b1, b2)

        sampled_1_cells = self.decoder_0_cells(embeddings_1, b10)
        sampled_2_cells = self.decoder_1_cells(embeddings_2, b20)

        return sampled_1_cells, sampled_2_cells

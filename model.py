import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from sampler import SampleIncidenceRow as SampleRow  # Import SampleRow from sampler.py
import numpy as np

class AutoregressiveSubsetSampler(nn.Module):
    def __init__(self, n, n_features, size_g, m_features, new_cell_factor, leaf_prob=0.5, left_prob=0.67, right_prob=0.67, min_nonzero=2, max_nonzero=6):
        super().__init__()
        self.n = n
        self.n_features = n_features
        self.size_g = size_g
        self.m_features = m_features
        self.leaf_prob = leaf_prob
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.min_nonzero = min_nonzero
        self.max_nonzero = max_nonzero
        self.new_cell_factor = new_cell_factor

        # MLP for processing m and g
        self.mlp_mg = nn.Sequential(
            nn.Linear(size_g + size_g, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )

        # SampleRow module
        self.sample_row = SampleRow(2, n, size_g, self.leaf_prob, self.left_prob, self.right_prob, 
                                    min_nonzero=self.min_nonzero, max_nonzero=self.max_nonzero)

        # Transformer for processing g
        self.transformer = nn.TransformerEncoderLayer(d_model=size_g, nhead=1)

    def forward(self, gnn_embeds, b):
        num_nodes = gnn_embeds.size(0)
        # g = torch.randn(self.size_g, device=gnn_embeds.device)    
        g = torch.zeros(self.size_g, device=gnn_embeds.device)
        # print("gnn_embeds", gnn_embeds.size())
        # print("b", b.size())


        b = b.to_dense()

        incidence_matrix = []


        
        padded_gnn_embeds_size = int(gnn_embeds.size(0)*self.new_cell_factor)
        padding_size = padded_gnn_embeds_size - gnn_embeds.size(0)
        padding = torch.zeros(padding_size, gnn_embeds.size(1), device=gnn_embeds.device)
        padded_gnn_embeds = torch.cat([gnn_embeds, padding], dim=0)
        
        print("generating rows")
        for i in tqdm(range(padded_gnn_embeds_size)):
            m = padded_gnn_embeds[i]
            h = self.mlp_mg(torch.cat([m, g]))


            valid_sample = False
            while not valid_sample:
                row_sample, g_new = self.sample_row(h)
                nonzero_count = row_sample.nonzero().size(0)
                if nonzero_count == 0 or (self.min_nonzero <= nonzero_count <= self.max_nonzero):
                    
                    valid_sample = True
            
            incidence_matrix.append(row_sample)
            g = self.transformer(g_new.unsqueeze(0)).squeeze(0)

        incidence_matrix = torch.stack(incidence_matrix)



        print("RETURNING INCIDENCE MATRIX")
        print(incidence_matrix)
        print("incidence matrix size was: ", b.size(0))
        print("new size: ", incidence_matrix.size(0))
        return incidence_matrix

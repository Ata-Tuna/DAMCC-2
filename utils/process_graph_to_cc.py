import sys
import os
import toponetx as tpx
import networkx as nx
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import argparse
import numpy as np

def ensure_nonzero_dim(tensor, axis=0):
    if tensor.shape[axis] == 0:
        shape = list(tensor.shape)
        shape[axis] = 1  # Set the 0-dimension to 1

        if axis == 0:
            zero_tensor = torch.zeros(1, shape[1], dtype=tensor.dtype).to_sparse()
        else:
            zero_tensor = torch.zeros(shape[0], 1, dtype=tensor.dtype).to_sparse()

        tensor = torch.cat([tensor, zero_tensor], dim=axis)
    
    return tensor

class CCDataset(Dataset):
    """Class for the SHREC 2016 dataset.

    Parameters
    ----------
    data : npz file
        npz file containing the SHREC 2016 data.
    """
     
    def __init__(self, data, test=False) -> None:
        self.data = data
        self.test = test
        self.complexes = self._graph_data_to_cc_data(data)
        self.a01, self.a02, self.coa2, self.b1, self.b2, self.b10, self.b20, self.b10_t, self.b20_t = self._get_neighborhood_matrix(self.complexes)
        self.x_0 = self._extract_x_0(data)
        self.x_1, self.x_2 = self._extract_x_1_x_2(self.complexes)

        self.a01 = self._flatten_list_of_lists(self.a01)
        self.a02 = self._flatten_list_of_lists(self.a02)
        self.coa2 = self._flatten_list_of_lists(self.coa2)
        self.b1 = self._flatten_list_of_lists(self.b1)
        self.b2 = self._flatten_list_of_lists(self.b2)
        self.b10 = self._flatten_list_of_lists(self.b10)
        self.b20 = self._flatten_list_of_lists(self.b20)
        self.b10_t = self._flatten_list_of_lists(self.b10_t)
        self.b20_t = self._flatten_list_of_lists(self.b20_t)
        self.x_0 = self._flatten_list_of_lists(self.x_0)
        self.x_1 = self._flatten_list_of_lists(self.x_1)
        self.x_2 = self._flatten_list_of_lists(self.x_2)
        self.complexes = self._flatten_list_of_lists(self.complexes)

        num_values = len(self.a01)
        random_binary_values = torch.randint(low=0, high=2, size=(num_values,))
        self.y = random_binary_values

    def _flatten_list_of_lists(self, list_of_lists):
        return [tensor for sublist in list_of_lists for tensor in sublist]

    def _convert_graph_to_cc_via_clique(self, G):
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))

        cc = tpx.transform.graph_to_simplicial_complex.graph_to_clique_complex(G)
        cc = cc.to_cell_complex()

        # This is to maaintain the order of the nodes and edges
        for edge in G.edges():
            cc.add_cell(edge, rank=1)
        for node in G.nodes():
            cc.add_node(node)
        cc = cc.to_combinatorial_complex()
        return cc

    def _graph_data_to_cc_data(self, graph_data):
        cc_data = []
        for graph_seq in graph_data:
            cc_seq = []
            print('Converting graphs to ccs')
            for graph in tqdm(graph_seq):
                cc = self._convert_graph_to_cc_via_clique(graph)
                cc_seq.append(cc)
            cc_data.append(cc_seq)
        return cc_data

    def _extract_x_0(self, graph_data):
        cc_data = []
        for graph_seq in graph_data:
            cc_seq = []
            print('Setting x_0')
            for graph in tqdm(graph_seq):
                G = graph
                if len(G.nodes) > 0 and len(list(G.nodes(data=True))[0][1]) > 0:
                    feature_matrix = torch.Tensor([list(G.nodes[node].values())[0] for node in G.nodes()])
                else:
                    feature_matrix = torch.eye(len(G.nodes))
                cc_seq.append(feature_matrix)
            if not self.test:
                cc_data.append(cc_seq[:-1])
            else:
                cc_data.append(cc_seq)
        return cc_data

    def _extract_x_1_x_2(self, complexes):
        x1batch = []
        x2batch = []
        for cc_seq in complexes:
            x1_seq = []
            x2_seq = []
            # print("Setting x_1_x_2")
            # print("cc_seq", (cc_seq[1]))
            for cc in tqdm(cc_seq):
                B2 = cc.incidence_matrix(1, 2, index=False)
                B2 = B2.todense()
                if B2.shape[1] == 0 and B2.shape[0] != 0:
                    dims = (B2.shape[0], 1)
                elif B2.shape[1] != 0 and B2.shape[0] == 0:
                    dims = (1, B2.shape[1])
                elif B2.shape[0] == 0 and B2.shape[1] == 0:
                    dims = (1, 1)
                else:
                    dims = B2.shape
                x1_feature_matrix = torch.ones(dims[0], 1)
                x2_feature_matrix = torch.ones(dims[1], 1)
                # print("x1_feature_matrix", x1_feature_matrix.shape)
                x1_seq.append(x1_feature_matrix)
                x2_seq.append(x2_feature_matrix)
            if not self.test:
                x1batch.append(x1_seq[:-1])
                x2batch.append(x2_seq[:-1])
            else:
                x1batch.append(x1_seq)
                x2batch.append(x2_seq)
        return x1batch, x2batch

    def _get_neighborhood_matrix(self, complexes):
        a01batch = []
        a02batch = []
        coa2batch = []
        b1batch = []
        b2batch = []
        cob01batch = []
        cob02batch = []
        target_cob01batch = []
        target_cob02batch = []
        for cc_seq in complexes:
            a01seq = []
            a02seq = []
            coa2seq = []
            b1seq = []
            b2seq = []
            cob01seq = []
            cob02seq = []
            target_cob01seq = []
            target_cob02seq = []
            for cc in cc_seq:
                a01 = torch.from_numpy(cc.adjacency_matrix(0, 1).todense()).to_sparse()
                a01 = ensure_nonzero_dim(a01, axis=0)
                a01 = ensure_nonzero_dim(a01, axis=1)
                a01seq.append(a01)

                a02 = torch.from_numpy(cc.adjacency_matrix(1, 2).todense()).to_sparse()
                a02 = ensure_nonzero_dim(a02, axis=0)
                a02 = ensure_nonzero_dim(a02, axis=1)
                a02seq.append(a02)

                B = cc.incidence_matrix(rank=1, to_rank=2)
                A = B.T @ B
                A.setdiag(0)
                coa2 = torch.from_numpy(A.todense()).to_sparse()
                coa2 = ensure_nonzero_dim(coa2, axis=0)
                coa2 = ensure_nonzero_dim(coa2, axis=1)
                coa2seq.append(coa2)

                b1 = torch.from_numpy(cc.incidence_matrix(0, 1).todense()).to_sparse()
                b1 = ensure_nonzero_dim(b1, axis=0)
                b1 = ensure_nonzero_dim(b1, axis=1)
                b1seq.append(b1)

                b2 = torch.from_numpy(cc.incidence_matrix(1, 2).todense()).to_sparse()
                b2 = ensure_nonzero_dim(b2, axis=0)
                b2 = ensure_nonzero_dim(b2, axis=1)
                b2seq.append(b2)

                cob01 = torch.from_numpy(cc.incidence_matrix(0, 1).todense().T).to_sparse()
                cob01 = ensure_nonzero_dim(cob01, axis=0)
                cob01 = ensure_nonzero_dim(cob01, axis=1)
                cob01seq.append(cob01)

                cob02 = torch.from_numpy(cc.incidence_matrix(0, 2).todense().T).to_sparse()
                cob02 = ensure_nonzero_dim(cob02, axis=0)
                cob02 = ensure_nonzero_dim(cob02, axis=1)
                cob02seq.append(cob02)

            if not self.test:
                target_cob01seq = cob01seq[1:]
                target_cob02seq = cob02seq[1:]

                a01seq = a01seq[:-1]
                a02seq = a02seq[:-1]
                coa2seq = coa2seq[:-1]
                b1seq = b1seq[:-1]
                b2seq = b2seq[:-1]
                cob01seq = cob01seq[:-1]
                cob02seq = cob02seq[:-1]
            else:
                target_cob01seq = cob01seq
                target_cob02seq = cob02seq

            a01batch.append(a01seq)
            a02batch.append(a02seq)
            coa2batch.append(coa2seq)
            b1batch.append(b1seq)
            b2batch.append(b2seq)
            cob01batch.append(cob01seq)
            cob02batch.append(cob02seq)
            target_cob01batch.append(target_cob01seq)
            target_cob02batch.append(target_cob02seq)

        return a01batch, a02batch, coa2batch, b1batch, b2batch, cob01batch, cob02batch, target_cob01batch, target_cob02batch

    def num_classes(self) -> int:
        return len(np.unique(self.y))

    def channels_dim(self) -> tuple[int, int, int]:
        return [self.x_0[0].shape[1], self.x_1[0].shape[1], self.x_2[0].shape[1]]

    def __len__(self) -> int:
        return len(self.complexes)

    def get_via_indices(self, id1x, id2x) -> tuple[torch.Tensor, ...]:
        return (
            self.x_0[id1x][id2x],
            self.a01[id1x][id2x],
            self.a02[id1x][id2x],
            self.b10[id1x][id2x],
            self.b20[id1x][id2x],
            self.b10_t[id1x][id2x],
            self.b20_t[id1x][id2x]
        )

    def __getitem__(self, id1x) -> tuple[torch.Tensor, ...]:
        return (
            self.x_0[id1x],
            self.x_1[id1x],
            self.x_2[id1x],
            self.a01[id1x],
            self.a02[id1x],
            self.coa2[id1x],
            self.b1[id1x],
            self.b2[id1x],
            self.b10[id1x],
            self.b20[id1x],
            self.b10_t[id1x],
            self.b20_t[id1x]
        )

def main(input_file, output_dir, test):
    with open(input_file, 'rb') as file:
        graph_data = pickle.load(file)

    cc_data = CCDataset(graph_data, test=test)

    base_name = os.path.basename(input_file)
    name_without_extension = os.path.splitext(base_name)[0]
    name_without_extension = name_without_extension.replace("graphs", "").replace("raw", "").strip("_")

    output_file_name = f"{name_without_extension}_ccs.pkl"
    output_file = os.path.join(output_dir, output_file_name)

    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    with open(output_file, 'wb') as file:
        pickle.dump(cc_data, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process graph data and save as CCDataset.")
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input pickle file containing graph data.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save the processed CCDataset.')
    parser.add_argument('-t', '--test', action='store_true', help='If set, bypass removal of first and last elements in sequences')
    args = parser.parse_args()

    main(args.input_path, args.output_dir, test=args.test)
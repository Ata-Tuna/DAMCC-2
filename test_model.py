import torch
import pickle
import os
import sys
from utils.utils import (incidence_matrix_to_graph, 
                    generate_cc_from_transposed_incidence, 
                    divide_tensors_into_lists,
                    divide_tensors)
from colorama import Fore, Style

ascii_banner_test = f"""
{Fore.RED}   ■  ▗▞▀▚▖ ▄▄▄  ■         ▐▌▗▞▀▜▌   ■  ▗▞▀▜▌     ▄▄▄ ▗▞▀▜▌▄   ▄ ▗▞▀▚▖   ▐▌
▗▄▟▙▄▖▐▛▀▀▘▀▄▄▗▄▟▙▄▖       ▐▌▝▚▄▟▌▗▄▟▙▄▖▝▚▄▟▌    ▀▄▄  ▝▚▄▟▌█   █ ▐▛▀▀▘   ▐▌
  ▐▌  ▝▚▄▄▖▄▄▄▀ ▐▌      ▗▞▀▜▌       ▐▌           ▄▄▄▀       ▀▄▀  ▝▚▄▄▖▗▞▀▜▌
  ▐▌            ▐▌      ▝▚▄▟▌       ▐▌                                ▝▚▄▟▌
  ▐▌            ▐▌                  ▐▌                                     
{Style.RESET_ALL}
"""

def test_model(model, num_nodes, test_loader, model_path, save_path, test_graphs_data, device):
    model.eval()
    model.load_state_dict(torch.load(model_path))  # Load the specific model for testing
    total_test_loss = 0
    sampled_b10_list = []
    sampled_b20_list = []
    target_b10_list = []
    target_b20_list = []
    sampled_graphs = []
    sampled_ccs = []

    try:
        with torch.no_grad():
            for cc in test_loader:
                x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, b10_t, b20_t  = [tensor.to(device) for tensor in cc]

                # Convert adjacency matrices to Float
                a1 = a1.float()
                a2 = a2.float()
                coa2 = coa2.float()
                b1 = b1.float()
                b2 = b2.float()

                sampled_b10, sampled_b20 = model(x_0, x_1, x_2, a1, a2, coa2, b1, b2, b10, b20, num_nodes)

                # Remove duplicate rows
                sampled_b20 = torch.unique(sampled_b20, dim=0)
                sampled_b10 = torch.unique(sampled_b10, dim=0)

                sampled_b10 = sampled_b10.cpu().numpy()
                sampled_b20 = sampled_b20.cpu().numpy()
                sampled_b10_list.append(sampled_b10)  # Convert tensors to numpy for saving
                sampled_b20_list.append(sampled_b20)
                target_b10_list.append(b10_t)  # Convert tensors to numpy for saving
                target_b20_list.append(b20_t)
                sampled_graphs.append(incidence_matrix_to_graph(sampled_b10))
                sampled_ccs.append(generate_cc_from_transposed_incidence(sampled_b10, sampled_b20))

        print("sampled_graphs: ",len(sampled_graphs))
        print("test_graphs_data: ",len(test_graphs_data[0]))
        # sys.exit()
        # Remove the first element from each inner list in test_graphs_data
        test_graphs_data = [inner_list[1:] for inner_list in test_graphs_data]
        # bring to the same shape as test data. This chunk is done for evaluation 
        if len(sampled_graphs) % len(test_graphs_data) != 0:
            raise ValueError("The number of sampled graphs must be divisible by the length of the test graphs data.")
        if len(sampled_ccs) % len(test_graphs_data) != 0:
            raise ValueError("The number of sampled CCs must be divisible by the length of the test graphs data.")
        sampled_graphs = divide_tensors_into_lists(test_graphs_data, sampled_graphs)
        sampled_ccs = divide_tensors_into_lists(test_graphs_data, sampled_ccs)
        sampled_b10s = divide_tensors(test_graphs_data, sampled_b10_list, sample=True)
        sampled_b20s = divide_tensors(test_graphs_data, sampled_b20_list, sample=True)
        target_b10s = divide_tensors(test_graphs_data, target_b10_list, sample=False)
        target_b20s = divide_tensors(test_graphs_data, target_b20_list, sample=False)

        # Save generated lists
        with open(os.path.join(save_path, 'target_b10s.pkl'), 'wb') as f:
            pickle.dump(target_b10s, f)
        with open(os.path.join(save_path, 'target_b20s.pkl'), 'wb') as f:
            pickle.dump(target_b20s, f)

        with open(os.path.join(save_path, 'sampled_b10s.pkl'), 'wb') as f:
            pickle.dump(sampled_b10s, f)
        with open(os.path.join(save_path, 'sampled_b20s.pkl'), 'wb') as f:
            pickle.dump(sampled_b20s, f)

        with open(os.path.join(save_path, 'sampled_graphs.pkl'), 'wb') as f:
            pickle.dump(sampled_graphs, f)
        with open(os.path.join(save_path, 'sampled_ccs.pkl'), 'wb') as f:
            pickle.dump(sampled_ccs, f)

        print(ascii_banner_test)
        print(f"Test results saved to {save_path}")
        
    except KeyboardInterrupt:
        print("Testing interrupted by user. Saving current state and exiting...")
        # Save the current state of the results
        with open(os.path.join(save_path, 'interrupted_sampled_b10_list.pkl'), 'wb') as f:
            pickle.dump(sampled_b10_list, f)
        with open(os.path.join(save_path, 'interrupted_sampled_b20_list.pkl'), 'wb') as f:
            pickle.dump(sampled_b20_list, f)
        with open(os.path.join(save_path, 'interrupted_sampled_graphs.pkl'), 'wb') as f:
            pickle.dump(sampled_graphs, f)
        with open(os.path.join(save_path, 'interrupted_sampled_ccs.pkl'), 'wb') as f:
            pickle.dump(sampled_ccs, f)
        print("Test results saved. Exiting gracefully.")
        sys.exit()
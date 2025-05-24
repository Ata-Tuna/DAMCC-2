import os
import sys
import torch
import pickle
import shutil
from datetime import datetime
from config import parse_arguments
from train_model import train_model, validate_model, save_training_state
from test_model import test_model
from damcc import Damcc
from toy_model import ToyModel
from loss_functions import sinkhorn_cosine_loss, wasserstein_l2_loss
from utils.process_graph_to_cc import CCDataset
import torch.optim as optim

# Parse arguments
args = parse_arguments()

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set default save directory if not provided
if args.save_dir is None:
    args.save_dir = 'experiments/last_experiment' if args.test else '/workspace/damcc/experiments'

# Create experiments directory if it doesn't exist
os.makedirs(args.save_dir, exist_ok=True)

# Model and Dataset information for experiment folder naming
model_name = 'damcc'
dataset_name = args.data_name
experiment_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
experiment_dir = os.path.join(args.save_dir, f"test" if args.test else f"{model_name}_{dataset_name}_{experiment_time}")
os.makedirs(experiment_dir, exist_ok=True)

# Initialize model parameters
n_of_ccs = 10000

if args.train:
    with open(args.train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(args.val_file, 'rb') as f:
        val_data = pickle.load(f)

    train_loader = [cc for i, cc in enumerate(train_data) if i < n_of_ccs]
    val_loader = [cc for i, cc in enumerate(val_data) if i < n_of_ccs]

if args.test:
    with open(args.test_file, 'rb') as f:
        test_data = pickle.load(f)
    test_loader = [cc for i, cc in enumerate(test_data) if i < n_of_ccs]

    with open(args.test_graphs_file, 'rb') as f:
        test_graphs_data = pickle.load(f)

# Extract basic model dimensions from the dataset
num_nodes = len(train_data.x_0[0]) if args.train else len(test_data.x_0[0])
n_features = len(train_data.x_0[0][1]) if args.train else len(test_data.x_0[0][1])
feature_n_0_cells = len(train_data.x_0[0][1]) if args.train else len(test_data.x_0[0][1])
feature_n_1_cells = 1
feature_n_2_cells = 1
size_g = 2


# Initialize model and optimizer
# model = Damcc(num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor = 1.5).to(device)
model = ToyModel(num_nodes, n_features, size_g, feature_n_0_cells, feature_n_1_cells, feature_n_2_cells, new_cell_factor = 1.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_function = sinkhorn_cosine_loss

# Early stopping and learning rate reduction
best_val_loss = float('inf')
lr_reduction_counter = 0
lr_reduction_patience = 10  # Reduce LR if no improvement after 10 epochs
early_stop_patience = 20    # Stop training if no improvement after 20 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_reduction_patience, factor=0.5, verbose=True)
one_cells_weight = 1



# Execute Training or Testing
if args.train:
    train_model(model, train_loader, val_loader, optimizer, 
                scheduler, loss_function, args.max_epochs, num_nodes, 
                device, experiment_dir, args, early_stop_patience, one_cells_weight
                )
elif args.test:
    if args.model_path is None:
        raise ValueError("Model path must be provided for testing.")
    test_model(model, num_nodes, test_loader, args.model_path, experiment_dir, test_graphs_data, device)
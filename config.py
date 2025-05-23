import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Testing Script")
    parser.add_argument('--data_name', type=str, help='name of the data for saving')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--train_file', type=str, help='Path to the training file')
    parser.add_argument('--val_file', type=str, help='Path to the validation file')
    parser.add_argument('--test_file', type=str, help='Path to the testing file')
    parser.add_argument('--test_graphs_file', type=str, help='Path to the testing graphs file')
    parser.add_argument('--model_path', type=str, help='Path to the best_val.pth model for testing')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save model and results')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    return parser.parse_args()
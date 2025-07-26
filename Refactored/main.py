import torch
from .hyperparameter_search import param_search, create_linear_param_iterator

# Set precision for better performance
torch.set_float32_matmul_precision('high')

# Run hyperparameter search
if __name__ == "__main__":
    param_search(create_linear_param_iterator(), summary_file="hp_search_linear.csv") 
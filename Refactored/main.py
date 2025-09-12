import os
import torch
from .hyperparameter_search import (
    param_search, create_linear_param_iterator, 
    create_one_hidden_layer_param_iterator, create_two_hidden_layer_param_iterator,
    evaluate_with_params
)
from .self_supervised_learning import SentenceTransformerSelfSupervised
from .config import datamodule_params
from .data_module import TweetsDataModule
import pytorch_lightning as pl

# Set precision for better performance
torch.set_float32_matmul_precision('high')

datamodule_params = {
    "batch_size": 64,
    "target_col": 'AR',
    "test_size": 0.2,
    "validation_size": 0.2,
    "oversample": True,
    "random_state": 2025
}

def continue_from_checkpoint(model: type[pl.LightningModule], checkpoint_path: str) -> pl.LightningModule:
    """Load model weights from a checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}. Starting from scratch.", flush=True)
        return None
    print(f"Loading model from checkpoint: {checkpoint_path}", flush=True)
    model_instance = model.load_from_checkpoint(checkpoint_path)
    return model_instance

# Run hyperparameter search
if __name__ == "__main__":
    # param_search(create_linear_param_iterator(), summary_file="hp_search_linear.csv") 
    # param_search(create_one_hidden_layer_param_iterator(), summary_file="hp_search_one_layer.csv") 
    # param_search(create_two_hidden_layer_param_iterator(), summary_file="hp_search_two_layer.csv") 
    model = continue_from_checkpoint(SentenceTransformerSelfSupervised, checkpoint_path="checkpoints/last-v2.ckpt")
    model_params = {
        "model": model,
        "use_soft_labels": True,
        "learning_rate": 2e-4,
        "freeze_encoder": True,
    }

    dataModule = TweetsDataModule.read_csv(**datamodule_params)
    dataModule.setup("fit")
    
    evaluate_with_params(dataModule, model_params)
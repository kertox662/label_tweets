import pytorch_lightning as pl
import torch
import os

def continue_from_checkpoint(model: type[pl.LightningModule], checkpoint_path: str) -> pl.LightningModule:
    """Load model weights from a checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}. Starting from scratch.", flush=True)
        return None
    print(f"Loading model from checkpoint: {checkpoint_path}", flush=True)
    model_instance = model.load_from_checkpoint(checkpoint_path)
    return model_instance

CHECKPOINT = "st-ssl-epoch=03-val_loss=0.032.ckpt"

if __name__ == "__main__":
    from Refactored.self_supervised_learning import SentenceTransformerSelfSupervised

    model = continue_from_checkpoint(SentenceTransformerSelfSupervised, CHECKPOINT)
    if model is not None:
        print("Model loaded successfully.")
        print(model)
    else:
        print("Failed to load model.")
    
    
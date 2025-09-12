"""Refactored/self_supervised_learning.py

Self-supervised training utilities built around `sentence-transformers`.

After training you can pass the *output directory* produced here to
`hyperparameter_search.py` via the `transformer_model_name` parameter (or just
add it to `config.MODEL_OPTIONS`).
"""
import os
from typing import List
# import pyreadr

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from sentence_transformers import SentenceTransformer, losses, InputExample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Refactored.data_module_unsupervised import TweetsDataModuleUnSupervised

class SentenceTransformerSelfSupervised(pl.LightningModule):
    """Fine-tunes a SentenceTransformer encoder with MultipleNegativesRankingLoss."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        learning_rate: float = 5e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Sentence-Transformer encoder
        self.model = SentenceTransformer(model_name)
        # InfoNCE-style loss implemented inside ST
        self.loss_fn = losses.MultipleNegativesRankingLoss(self.model, scale=7)

    # def forward(self, sentences: List[str]):  # type: ignore[override]
    #     """Embed sentences using the underlying ST model."""
        # return self.model.encode(sentences, convert_to_tensor=True, device=self.device)
    def forward(self, features):
        """Forward pass through the SentenceTransformer backbone."""
        # pass1 = self.model(features)
        # pass2 = self.model(features)
        # print("Forward:", pass1.shape, pass2.shape, type(pass1), type(pass2), flush=True)
        # return torch.stack([pass1, pass2], dim=0)
        return self.model(features)

    def training_step(self, batch, batch_idx):
        # The smart batching collate returns (sentence_features, labels)
        sentence_features, labels = batch
        # embeddings = self(sentence_features)
        self.model.train()
        loss = self.loss_fn(sentence_features, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # Validation is optional for unsupervised SimCSE; we reuse the same loss
    def validation_step(self, batch, batch_idx):  # noqa: D401
        sentence_features, labels = batch
        self.model.train()
        loss = self.loss_fn(sentence_features, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def save_encoder(self, output_dir: str) -> str:
        """Save the fine-tuned SentenceTransformer and return the directory path."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        return output_dir
    
    def get_sentence_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

def continue_from_checkpoint(model: type[pl.LightningModule], checkpoint_path: str) -> pl.LightningModule:
    """Load model weights from a checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}. Starting from scratch.", flush=True)
        return None
    print(f"Loading model from checkpoint: {checkpoint_path}", flush=True)
    model_instance = model.load_from_checkpoint(checkpoint_path)
    return model_instance


def pre_train_sentence_transformer(
    data_path: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    max_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    save_path: str = "pre_trained_sentence_st",
    is_checkpoint: bool = False,
):
    """Fine-tune *model_name* on *data_path* and save the encoder to *save_path*."""

    if is_checkpoint:
        model = continue_from_checkpoint(SentenceTransformerSelfSupervised, checkpoint_path=model_name)
    else:
        model = SentenceTransformerSelfSupervised(model_name=model_name, learning_rate=learning_rate)
    print("Model Created========================================")
    data = pd.read_csv(data_path)
    print(f"Data Read========================================")
    # total_gpus = torch.cuda.device_count()
    # max_total_workers = os.cpu_count() or 1
    # num_workers = max(1, max_total_workers // total_gpus)
    datamodule = TweetsDataModuleUnSupervised(data=data, batch_size=batch_size, num_workers=1)
    print("Data Module Created ========================================")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=4)
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        save_last= True,
        filename="st-ssl-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
        every_n_epochs=2,
    )

    # ------------------------------------------------------------
    # Build Trainer dynamically so the same script works on CPU,
    # single-GPU, and multi-GPU (DDP) setups.
    # ------------------------------------------------------------
    cuda_gpus = torch.cuda.device_count()
    mps_available = torch.backends.mps.is_available() and not torch.cuda.is_available()
    max_steps = 1000000
    # val_check_interval = 5000
    val_check_interval = float(0.25)  # check 4 times per epoch
    trainer_args = dict(
        # max_steps=max_steps,
        max_epochs=max_epochs,
        val_check_interval = val_check_interval,
        callbacks=[early_stop, ckpt],
        logger=TensorBoardLogger("tb_logs", name="st_ssl"),
        log_every_n_steps=200,
        accumulate_grad_batches=8,
    )

    if cuda_gpus:
        trainer_args.update(dict(precision="16-mixed"))
        if cuda_gpus == 1:
            print("Single GPU ============================")
            trainer_args.update(dict(accelerator="gpu", devices=1))
        else:
            print("Multiple GPU ============================")
            trainer_args.update(
                dict(
                    accelerator="gpu",
                    devices=cuda_gpus,
                    strategy=DDPStrategy(find_unused_parameters=True),
                )
            )
    elif mps_available:
        # Apple-silicon Metal Performance Shaders backend
        trainer_args.update(dict(accelerator="mps", devices=1, precision="16-mixed"))
    else:
        trainer_args.update(dict(accelerator="cpu", devices=1, precision=32))

    trainer = pl.Trainer(**trainer_args)
    print("Trainer Fitting ========================================")
    trainer.fit(model, datamodule)

    # Save the fine-tuned encoder for downstream tasks
    output_dir = model.save_encoder(save_path)
    print(f"Self-supervised model saved to: {output_dir}")
    return output_dir




if __name__ == "__main__":
    # r_data = pyreadr.read_r('Refactored/data/raw_tweets.Rdata')
    # df = r_data['raw_tweets'].sample(1000)
    # # df.to_csv("sample_r_data.csv")
    torch.set_float32_matmul_precision('medium')
    pre_train_sentence_transformer(
        data_path="Refactored/data/raw_tweets.csv", max_epochs=10,
        model_name="checkpoints/st-ssl-epoch=05-val_loss=0.031.ckpt", is_checkpoint=True,
        )

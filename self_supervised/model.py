import os
from typing import Optional
import torch
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer, losses

TRAIN_LOSS = "train_loss"
VALIDATION_LOSS = "val_loss"

class TransformerBackbone(pl.LightningModule):
    """Fine-tunes a SentenceTransformer encoder with MultipleNegativesRankingLoss."""

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2",
        learning_rate: float = 5e-5,
        loss_scale: float = 7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Sentence-Transformer encoder
        self.model = SentenceTransformer(model_name)
        # InfoNCE-style loss implemented inside ST
        self.loss_fn = losses.MultipleNegativesRankingLoss(self.model, scale=self.hparams.loss_scale)

    def forward(self, features):
        return self.model(features)

    def do_step(self, batch, loss_type):
        sentence_features, labels = batch
        loss = self.loss_fn(sentence_features, labels)
        self.log(loss_type, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, TRAIN_LOSS)

    # Validation is optional for unsupervised SimCSE; we reuse the same loss
    def validation_step(self, batch, batch_idx):  # noqa: D401
        self.train()
        with torch.no_grad():
            loss = self.do_step(batch, VALIDATION_LOSS)
        self.eval()
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def save_encoder(self, output_dir: str) -> str:
        """Save the fine-tuned SentenceTransformer and return the directory path."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        return output_dir

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup

class BertweetClassifier(pl.LightningModule):
    def __init__(
        self,
        transformer_model_name: str = "vinai/bertweet-base",
        num_labels: int = 3,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.10,
        freeze_encoder: bool = True,
        use_soft_labels: bool = False,
        classifier_constructor = None, # function (embedding_dim: int, num_labels: int) -> nn.Module
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.use_soft_labels = use_soft_labels
        # 1) encoder
        self.encoder = SentenceTransformer(transformer_model_name)

        # 2) Set the classifier that will run on the embedding
        #    If it is provided, just use that, otherwise make
        #    a simple linear layer.
        hidden = self.encoder.get_sentence_embedding_dimension()
        if classifier_constructor is not None:
            self.classifier = classifier_constructor(hidden, num_labels)
        else:
            self.classifier = nn.Linear(hidden, num_labels)

        # 3) (optionally) freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        cls_weights = None if class_weights is None else class_weights.to(self.device)
        if self.use_soft_labels:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=cls_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=cls_weights)

    # Forward expects ONLY the list[str] texts
    def forward(self, texts: list[str]):
        embeds = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        return self.classifier(embeds)

    # ------------------------------------------------------------------
    # Shared step to avoid duplication
    def _shared_step(self, batch, stage: str):
        texts, labels, soft_labels, true_AR, true_MB = batch
        logits = self(texts)
        if self.use_soft_labels:
            loss = self.loss_fn(logits, soft_labels.to(self.device))
        else:
            loss = self.loss_fn(logits, labels.to(self.device))
        
        # Get probabilities
        preds = logits.argmax(1)

        acc_AR = (preds == true_AR.to(self.device)).float().mean()
        acc_MB = (preds == true_MB.to(self.device)).float().mean()


        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        self.log(f"{stage}_acc_AR", acc_AR, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc_MB", acc_MB, prog_bar=True, on_epoch=True, on_step=False)

        # Optionally: return predictions for logging or downstream use
        # Return as a dict for Lightning
        return {
            "loss": loss,
            "preds": preds,
            "true_AR": true_AR,
            "true_MB": true_MB,
        }

    # Lightning API -----------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        # Re-use the same bookkeeping you used for val:
        self._shared_step(batch, stage="test")
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # `batch` is just a list[str] from your predict_dataloader
        texts = batch
        logits = self(texts)
        probs  = torch.softmax(logits, dim=-1)
        return probs  

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            "optimizer":  optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",      # step every batch
                "frequency": 1,
                "name": "linear-warmup",
            },
        }
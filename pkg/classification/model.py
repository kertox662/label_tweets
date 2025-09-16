import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup

from self_supervised.model import TransformerBackbone

NUM_LABELS = 3

class BertweetClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        model_checkpoint: str | None = None,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.10,
        freeze_encoder: bool = True,
        use_soft_labels: bool = False,
        class_weights: torch.Tensor = None,

        hidden_dim: int = 128,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.use_soft_labels = use_soft_labels

        if model_checkpoint:
            self.encoder = TransformerBackbone.load_from_checkpoint(model_checkpoint).model
        else:
            self.encoder = SentenceTransformer(model_name)

        # 2) Set the classifier that will run on the embedding
        #    If it is provided, just use that, otherwise make
        #    a simple linear layer.
        embedding_dim = self.encoder.get_sentence_embedding_dimension()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, NUM_LABELS)
        )

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
        acc_ar = (preds == true_AR.to(self.device)).float().mean()
        acc_mb = (preds == true_MB.to(self.device)).float().mean()

        f1_ar = f1_score(true_AR.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        f1_mb = f1_score(true_MB.cpu().numpy(), preds.cpu().numpy(), average='weighted')

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        self.log(f"{stage}_acc_ar", acc_ar, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1_ar", f1_ar, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc_mb", acc_mb, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_f1_mb", f1_mb, prog_bar=True, on_epoch=True, on_step=False)

        # Return as a dict for Lightning
        return {
            "loss": loss,
            "preds": preds,
            "true_AR": true_AR,
            "true_MB": true_MB,
            "acc_ar": acc_ar,
            "acc_mb": acc_mb,
            "f1_ar": f1_ar,
            "f1_mb": f1_mb,
        }

    # Lightning API -----------------------------------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, stage="test")
        # Append for use in on_test_epoch_end
        self.test_outputs.append({
            "preds": result["preds"].cpu(),
            "true_AR": result["true_AR"].cpu(),
            "true_MB": result["true_MB"].cpu(),
            "acc_ar": result["acc_ar"],
            "acc_mb": result["acc_mb"],
            "f1_ar": result["f1_ar"],
            "f1_mb": result["f1_mb"],
        })
        return result

    def on_test_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.test_outputs])
        true_AR = torch.cat([x["true_AR"] for x in self.test_outputs])
        true_MB = torch.cat([x["true_MB"] for x in self.test_outputs])

        acc_ar = (preds == true_AR).float().mean().item()
        acc_mb = (preds == true_MB).float().mean().item()
        f1_ar = f1_score(true_AR.numpy(), preds.numpy(), average='weighted')
        f1_mb = f1_score(true_MB.numpy(), preds.numpy(), average='weighted')

        self.log("test_acc_ar", acc_ar)
        self.log("test_acc_mb", acc_mb)
        self.log("test_f1_ar", f1_ar)
        self.log("test_f1_mb", f1_mb)
        
        # Save results for outside access if needed
        self.test_metrics = {
            "test_acc_ar": acc_ar,
            "test_acc_mb": acc_mb,
            "test_f1_ar": f1_ar,
            "test_f1_mb": f1_mb,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # `batch` is just a list[str] from your predict_dataloader
        texts = batch
        logits = self(texts)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",      # step every batch
                "frequency": 1,
                "name": "linear-warmup",
            },
        }
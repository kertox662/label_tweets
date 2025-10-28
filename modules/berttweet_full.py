import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List


import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from torchinfo import summary

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassConfusionMatrix
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_NAME = "vinai/bertweet-base"

@dataclass
class TweetExample:
    text: str
    label: int
    

class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, text_col: str, label_col: str, max_length: int = 128):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.labels = [l - 1 for l in self.labels] # 0-indexed
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch: List[dict]):
        # dynamic padding
        return self.tokenizer.pad(batch, return_tensors="pt")
    
class TweetsTVTDataModule(pl.LightningDataModule):
    """
    Creates train/val/test data loaders
    """
    
    def __init__(self,
        data_path: str,
        text_col: str,
        label_col: str,
        label_names: List[str],
        num_labels: int,
        val_size: float = 0.15,
        test_size: float = 0.15,
        val_seed: int = 42,
        test_seed: int = 42,
        batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_path = data_path
        self.text_col = text_col
        self.label_col = label_col
        self.label_names = label_names
        self.num_labels = num_labels
        self.val_size = val_size
        self.test_size = test_size
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, normalization=True)
        self.collator = DataCollator(self.tokenizer)
    
    def read_csv(self, remove_disagreements: bool = False):
        data_df = pd.read_csv(self.data_path)
        data_df['AR'].replace({4: 2}, inplace=True)
        data_df['MB'].replace({4: 2}, inplace=True)

        if remove_disagreements:
            data_df = data_df[data_df["AR"] == data_df["MB"]]

        return data_df
        
    def setup(self, stage: Optional[str] = None):
        
        df = self.read_csv(False)

        assert self.text_col in df.columns, f"Missing text column {self.text_col}"
        assert self.label_col in df.columns, f"Missing label column {self.label_col}"

        # Stratified split if possible
        modeldev_df, test_df = train_test_split(
            df[[self.text_col, self.label_col]],
            test_size=self.test_size,
            random_state=self.test_seed,
            # stratify=df["AR"] if self.label_col in df and df["AR"].nunique() > 1 else None,
            stratify=df[self.label_col] if self.label_col in df and df[self.label_col].nunique() > 1 else None,
        )
        train_df, val_df = train_test_split(
            modeldev_df[[self.text_col, self.label_col]],
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.val_seed,
            stratify=modeldev_df[self.label_col] if self.label_col in modeldev_df and modeldev_df[self.label_col].nunique() > 1 else None,
        )
        self.train_ds = TweetDataset(train_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
        self.val_ds = TweetDataset(val_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
        self.test_ds = TweetDataset(test_df, self.tokenizer, self.text_col, self.label_col, self.max_length)

        # Save label counts for class weights
        self.train_label_counts = train_df[self.label_col].value_counts().reindex(range(1, self.num_labels+1), fill_value=0).sort_index().values


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)    
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True) 
    

class BertweetModule(pl.LightningModule):
    
    def __init__(self,
        num_labels: int,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        dropout: float = 0.1,
        total_steps: Optional[int] = None,
        class_weight: bool = False,
        train_label_counts: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_label_counts"]) # avoid logging large arrays
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        self.model.dropout = torch.nn.Dropout(p=dropout)

        # Metrics
        self.acc = MulticlassAccuracy(num_classes=num_labels)
        self.f1_macro = MulticlassF1Score(num_classes=num_labels, average='macro')
        self.f1_weighted = MulticlassF1Score(num_classes=num_labels, average='weighted')
        self.cm = MulticlassConfusionMatrix(num_classes=num_labels)

        # Optional class weights
        if class_weight and train_label_counts is not None:
            # inverse frequency weights; avoid div by zero
            counts = torch.tensor(train_label_counts, dtype=torch.float)
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(counts)
            self.register_buffer("class_weights", weights)
        else:
            self.class_weights = None
            
        self.total_steps_override = total_steps
        # print("Bertweet module hyperparameters:\n", self.hparams)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        if self.class_weights is not None:
            # recompute CE with weights (HF returns average CE w/o weights)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        # reset confusion matrix aggregator
        self.cm.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        y = batch["labels"]
        acc = self.acc(preds, y)
        f1_macro = self.f1_macro(preds, y)
        f1_weighted = self.f1_weighted(preds, y)
        cm = self.cm(preds, y)
        loss = outputs.loss
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits, y)
        self.log_dict({"val_loss": loss, "val_acc": acc, "val_macro_f1": f1_macro, "val_weighted_f1": f1_weighted}, prog_bar=True)
        return {"cm": cm}

    def on_validation_epoch_end(self):
        # compute and log confusion matrix as a TensorBoard figure
        try:
            cm = self.cm.compute().cpu().numpy()
            fig = self._plot_confusion_matrix(cm)
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_figure"):
                self.logger.experiment.add_figure("val_confusion_matrix", fig, global_step=self.current_epoch)
        except Exception as e:
            self.print(f"[warn] failed to log confusion matrix: {e}")
            
    def on_test_epoch_start(self):
        self.cm.reset()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        y = batch["labels"]
        acc = self.acc(preds, y)
        f1_macro = self.f1_macro(preds, y)
        f1_weighted = self.f1_weighted(preds, y)
        cm = self.cm(preds, y)
        loss = outputs.loss
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits, y)
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_macro_f1": f1_macro, "test_weighted_f1": f1_weighted}, prog_bar=True)
        return {"cm": cm}

    def on_test_epoch_end(self):
        # compute and log confusion matrix as a TensorBoard figure
        try:
            cm = self.cm.compute().cpu().numpy()
            fig = self._plot_confusion_matrix(cm)
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_figure"):
                self.logger.experiment.add_figure("test_confusion_matrix", fig, global_step=self.current_epoch)
        except Exception as e:
            self.print(f"[warn] failed to log confusion matrix: {e}")
            
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Standard forward pass; no labels needed
        outputs = self(**{k: v for k, v in batch.items() if k != "labels"})
        logits = outputs.logits
        return {"logits": logits.detach().cpu()}
    
    def _plot_confusion_matrix(self, cm):
        num_classes = cm.shape[0]
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            label_names = dm.label_names
        else:
            label_names = [f"Class {i}" for i in range(num_classes)]
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("True", fontsize=14)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(label_names, fontsize=12, rotation=45, ha="right")
        ax.set_yticklabels(label_names, fontsize=12)
        fig.colorbar(im, ax=ax)
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        
        # Scheduler with warmup
        if self.total_steps_override is None:
            # Will be set in trainer via setup hook
            return optimizer
        warmup_steps = int(self.hparams.warmup_ratio * self.total_steps_override)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.total_steps_override
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        # If total steps isn't set, try to infer from trainer
        if self.total_steps_override is None and self.trainer is not None:
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
            else:
                # steps per epoch * max_epochs
                train_loader = self.trainer.datamodule.train_dataloader()
                effective_bs = self.trainer.accumulate_grad_batches * (self.trainer.num_devices if isinstance(self.trainer.num_devices, int) else 1)
                steps_per_epoch = math.ceil(len(train_loader.dataset) / (train_loader.batch_size * effective_bs))
                total_steps = steps_per_epoch * self.trainer.max_epochs
            self.total_steps_override = total_steps 
            
        # ---- torchinfo summary (runs once, early in first setup) ----
        if not hasattr(self, "_printed_summary"):
            try:
                dm = getattr(self.trainer, "datamodule", None)
                if dm is not None and hasattr(dm, "tokenizer"):
                    tokenizer = dm.tokenizer
                    max_length = getattr(dm, "max_length", 128)
                    dummy = tokenizer(
                        ["hello twitter!", "torchinfo summary"],
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    # move to the current device to avoid device mismatch
                    dummy = {k: v.to(self.device) for k, v in dummy.items() if k in ("input_ids", "attention_mask")}
                    # Print the architecture summary
                    print("==== Summary for LightningBertweet ====")
                    print(summary(self.model, input_data=dummy, depth=2))
                    print("==== End Summary ====")
                    self._printed_summary = True
            except Exception as e:
                self.print(f"[warn] torchinfo summary failed: {e}")   
 
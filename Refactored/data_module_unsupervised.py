import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

from .preprocessing import preprocess_text
from sentence_transformers import InputExample, SentenceTransformer


class TweetsDataModuleUnSupervised(pl.LightningDataModule):
    def __init__(self, data: pd.DataFrame, batch_size: int = None,
                validation_size=0.1, oversample=False, random_state=2025,
                 model_name: str =  "/home/atjhin/projects/def-jhoey/atjhin/model/all-mpnet-base-v2",
                 num_workers=1):
        super().__init__()
        self.data = data.copy()
        self.data = self.data
        self.batch_size = batch_size
        # Auto-select workers: use all logical CPU cores unless overridden
        default_workers = os.cpu_count() or 1
        self.num_workers = default_workers if num_workers is None else num_workers

        # On macOS with MPS backend, enforce single-process dataloader to avoid pickling GPU tensors
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            self.num_workers = 0

        self._st_model = SentenceTransformer(model_name)

        self._validation_size = validation_size
        self._oversample = oversample
        self._random_state = random_state
    
    @classmethod
    def read_r(cls, filename: str = "Refactored/data/sample_r_data.csv",
                 batch_size: int = None, validation_size=0.1,
                 oversample=False, random_state=2025):
        tweets_labeled = pd.read_csv(filename)

        return cls(
            tweets_labeled, 
            batch_size, 
            validation_size,
            oversample,
            random_state
        )

    def setup(self, stage: str | None = None):
        self.data = preprocess_text(self.data, text_col="text")
        self.num_labels = len(self.data)
        self._train_test_split()
        
    def _train_test_split(self):
        self.train, self.val = train_test_split(self.data, train_size=1-self._validation_size, random_state=self._random_state, shuffle=True)
        
        print(f"Shape of training/validation: {self.train.shape[0]}/{self.val.shape[0]}")
    
    def _dataloader(self, dataset, shuffle, drop_last):
        examples = [InputExample(texts=[t, t]) for t in dataset["clean_text"]]
        return DataLoader(
            examples,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self._st_model.smart_batching_collate,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dataloader(dataset=self.train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._dataloader(dataset=self.val, shuffle=False, drop_last=False)
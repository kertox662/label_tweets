import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import pytorch_lightning as pl

from data.preprocessing import preprocess_text


class TweetsDataModule(pl.LightningDataModule):
    def __init__(self, data: pd.DataFrame, batch_size: int = None, target_col: str = 'AR',
                 test_size=0.2, validation_size=0.2, oversample=False, random_state=2025):
        super().__init__()
        self.data = data.copy()
        self.batch_size = batch_size
        self.target_col = target_col

        self.label2id: dict[str, int] = {}
        self.id2label: list[str] = []

        self._test_size = test_size
        self._validation_size = validation_size
        self._oversample = oversample
        self._random_state = random_state
    
    @classmethod
    def read_csv(cls, filename: str = "data/train_master.csv", remove_disagreements: bool = False,
                 batch_size: int = None, target_col: str = 'AR', test_size=0.2, validation_size=0.2,
                 oversample=False, random_state=2025):
        tweets_labeled = pd.read_csv(filename)
        tweets_labeled['AR'].replace({4: 2}, inplace=True)
        tweets_labeled['MB'].replace({4: 2}, inplace=True)

        if remove_disagreements:
            tweets_labeled = tweets_labeled[tweets_labeled["AR"] == tweets_labeled["MB"]]

        return cls(
            tweets_labeled, batch_size, target_col, test_size, validation_size,
            oversample, random_state
        )

    def setup(self, stage: str = None):
        if stage != "fit":
            return
        
        # Assign train/val datasets for use in dataloaders
        self.data = preprocess_text(self.data, text_col="text")
        uniques = sorted(self.data[self.target_col].unique())
        self.num_labels = len(uniques)
        self.label2id = {lbl: i for i, lbl in enumerate(uniques)}
        self.id2label = uniques                            # same order
        self.data["y"] = self.data[self.target_col].map(self.label2id)    
        self.data['AR_id'] = self.data['AR'].map(self.label2id)
        self.data['MB_id'] = self.data['MB'].map(self.label2id)
        self.data['soft_label'] = self.data.apply(self.make_soft_label, axis=1)
        self._train_test_split()
        
    def _train_test_split(self):
        train, self.test = train_test_split(self.data, train_size=1-self._test_size, random_state=self._random_state, shuffle=True)
        self.train, self.val = train_test_split(train, train_size=1-self._validation_size, random_state=self._random_state, shuffle=True)
        if self._oversample:
            ros = RandomOverSampler(random_state=self._random_state)
            # Extract features and label
            X_res, y_res = ros.fit_resample(self.train, self.train["y"])
            self.train = X_res.copy()
            self.train["y"] = y_res
        
        print(f"Shape of training/validation/test: {self.train.shape[0]}/{self.val.shape[0]}/{self.test.shape[0]}")
        print(f"Label distribution for training data: {self.train.y.value_counts()}")

    def make_soft_label(self, row):
        arr = np.zeros(self.num_labels)
        arr[self.label2id[row['AR']]] += 1
        arr[self.label2id[row['MB']]] += 1
        arr /= arr.sum()  # divide by 2
        return arr

    def _zip(self, data):
        return list(zip(
            data["clean_text"], 
            data["y"],
            data["soft_label"],
            data["AR_id"], 
            data["MB_id"]
        ))
    
    def _dataloader(self, dataset, shuffle, drop_last):
        data = self._zip(dataset)
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: (
                [t for t, _, _, _, _ in batch],
                torch.tensor([y for _, y, _, _, _ in batch], dtype=torch.long),
                torch.tensor([l for _, _, l, _, _ in batch], dtype=torch.float32),
                torch.tensor([ar for _, _, _, ar, _ in batch], dtype=torch.long),
                torch.tensor([mb for _, _, _, _, mb in batch], dtype=torch.long),
            ),
            drop_last=drop_last,
        )

    def train_dataloader(self):
        return self._dataloader(dataset=self.train, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._dataloader(dataset=self.val, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._dataloader(dataset=self.test, shuffle=False, drop_last=False)
    
    def predict_dataloader(self):
        """
        Returns a DataLoader for inference.
        - If you called `setup("predict")`, it will use `self.test`.
        - If you created `self.predict` manually (e.g., a new DataFrame without labels),
          that takes precedence.
        Collate fn yields a *list[str]* so the model's forward just receives raw texts.
        """
        dataset = getattr(self, "predict", None)
        if dataset is None:          # fall back to the held-out test set
            dataset = self.test

        texts = list(dataset["clean_text"])

        return DataLoader(
            texts,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # returns List[str] per batch
        )
    
    def train_class_weights(self):
        labels = np.array(self.train["y"])
        weights = compute_class_weight('balanced', classes=np.arange(3), y=labels)
        return torch.Tensor(weights) 
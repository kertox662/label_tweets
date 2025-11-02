from dataclasses import dataclass
from typing import Optional, List


import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

MODEL_NAME = "vinai/bertweet-base"

@dataclass
class TweetExample:
    text: str
    label: int
    

class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, text_col: str, label_col: str, max_length: int = 128, use_sentence_transformer = False):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.labels = [l - 1 for l in self.labels] # 0-indexed
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.label_col = label_col
        # self.use_sentence_transformer = use_sentence_transformer


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
            # add_special_tokens=True,
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
        model_name: str = MODEL_NAME,
        val_size: float = 0.15,
        test_size: float = 0.15,
        val_seed: int = 42,
        test_seed: int = 42,
        batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 2,
        test_only: bool = False,
        no_test: bool = False,
        remove_disagreements: bool = False,
        add_disgreed_to_test: bool = False,
        **kwargs
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
        self.test_only = test_only
        self.no_test = no_test
        self.remove_disagreements = remove_disagreements
        self.add_disgreed_to_test = add_disgreed_to_test
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
        # self.tokenizer.add_special_tokens({'additional_special_tokens': ['[T_LINK]', '[T_USER]']})
        self.collator = DataCollator(self.tokenizer)
    
    def read_csv(self):
        data_df = pd.read_csv(self.data_path)
        data_df['AR'].replace({4: 2}, inplace=True)
        data_df['MB'].replace({4: 2}, inplace=True)

        if self.remove_disagreements:
            self.disagreed_df = data_df[data_df["AR"] != data_df["MB"]]
            data_df = data_df[data_df["AR"] == data_df["MB"]]
        else:
            self.disagreed_df = None

        return data_df
        
    def setup(self, stage: Optional[str] = None):
        
        df = self.read_csv()

        assert self.text_col in df.columns, f"Missing text column {self.text_col}"
        assert self.label_col in df.columns, f"Missing label column {self.label_col}"
        
        if self.no_test:
            train_df, val_df = train_test_split(
                df[[self.text_col, self.label_col]],
                test_size=self.val_size,
                random_state=self.val_seed,
                stratify=df[self.label_col] if self.label_col in df and df[self.label_col].nunique() > 1 else None,
            )

            self.test_ds = None
            self.train_ds = TweetDataset(train_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
            self.val_ds = TweetDataset(val_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
            self.train_label_counts = self.train_label_counts = train_df[self.label_col].value_counts().reindex(range(1, self.num_labels+1), fill_value=0).sort_index().values
        elif self.test_only:
            # Load all data as test set
            self.test_ds = TweetDataset(df[[self.text_col, self.label_col]], self.tokenizer, self.text_col, self.label_col, self.max_length)
            self.train_ds = None
            self.val_ds = None
            self.train_label_counts = None
        else:
            print("Splitting data into train/val/test...")
            # Stratified split if possible
            modeldev_df, test_df = train_test_split(
                df[[self.text_col, self.label_col]],
                test_size=self.test_size,
                random_state=self.test_seed,
                stratify=df["AR"] if "AR" in df and df["AR"].nunique() > 1 else None,
                # stratify=df[self.label_col] if self.label_col in df and df[self.label_col].nunique() > 1 else None,
            )
            train_df, val_df = train_test_split(
                modeldev_df[[self.text_col, self.label_col]],
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.val_seed,
                stratify=modeldev_df[self.label_col] if self.label_col in modeldev_df and modeldev_df[self.label_col].nunique() > 1 else None,
            )

            if self.add_disgreed_to_test and self.disagreed_df is not None:
                test_df = pd.concat([ test_df, self.disagreed_df[[self.text_col, self.label_col]] ])

            self.train_ds = TweetDataset(train_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
            self.val_ds = TweetDataset(val_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
            self.test_ds = TweetDataset(test_df, self.tokenizer, self.text_col, self.label_col, self.max_length)

            # Save label counts for class weights
            self.train_label_counts = train_df[self.label_col].value_counts().reindex(range(1, self.num_labels+1), fill_value=0).sort_index().values


    def train_dataloader(self):
        if self.test_only:
            return None
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        if self.test_only:
            return None
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)    
    
    def test_dataloader(self):
        if self.no_test:
            return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)

    def predict_dataloader(self):
        if self.no_test:
            return None
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)


class KFoldDataModule(pl.LightningDataModule):
    """
    Custom data module for k-fold cross-validation that uses pre-split data
    """
    def __init__(self, train_df, val_df, test_df, tokenizer, collator, **kwargs):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        self.tokenizer = tokenizer
        self.collator = collator
        
        # Set attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def setup(self, stage: Optional[str] = None):
        # Override setup to use pre-split data instead of splitting again
        self.train_ds = TweetDataset(self._train_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
        self.val_ds = TweetDataset(self._val_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
        self.test_ds = TweetDataset(self._test_df, self.tokenizer, self.text_col, self.label_col, self.max_length)
        
        # Save label counts for class weights
        self.train_label_counts = self._train_df[self.label_col].value_counts().reindex(range(self.num_labels), fill_value=0).sort_index().values
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)    
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collator, persistent_workers=True)


def create_k_fold_data_modules(
    data_path: str,
    text_col: str,
    label_col: str,
    label_names: List[str],
    num_labels: int,
    model_name: str = MODEL_NAME,
    num_folds: int = 5,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 2,
    val_size: float = 0.15,
    random_state: int = 42,
    remove_disagreements: bool = False
):
    """
    Create k-fold cross-validation data modules with stratified sampling.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        text_col (str): Name of the column containing text data
        label_col (str): Name of the column containing labels
        label_names (List[str]): Names of the label classes
        num_labels (int): Number of label classes
        k (int): Number of folds for cross-validation
        batch_size (int): Batch size for dataloaders
        max_length (int): Maximum sequence length for tokenization
        num_workers (int): Number of workers for dataloaders
        test_size (float): Proportion of data to use for test set
        random_state (int): Random seed for reproducibility
        remove_disagreements (bool): Whether to remove rows where AR != MB
        
    Yields:
        tuple: (fold_idx, data_module) for each fold
    """
    # Read and preprocess data
    data_df = pd.read_csv(data_path)
    data_df['AR'].replace({4: 2}, inplace=True)
    data_df['MB'].replace({4: 2}, inplace=True)
    
    if remove_disagreements:
        data_df = data_df[data_df["AR"] == data_df["MB"]]
    
    # Prepare features and labels for k-fold on all data
    X = data_df[[text_col, label_col]]
    y = data_df[label_col]
    
    # Initialize tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
    collator = DataCollator(tokenizer)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    # Generate folds
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        # First split: separate test set for this fold
        fold_train_val_df = data_df.iloc[train_val_idx].reset_index(drop=True)
        test_df = data_df.iloc[test_idx].reset_index(drop=True)
        
        # Second split: train/val split on the remaining data
        train_df, val_df = train_test_split(
            fold_train_val_df[[text_col, label_col]],
            test_size=val_size / (1 - val_size),  # Adjust test_size for remaining data
            random_state=random_state,
            stratify=fold_train_val_df[label_col] if fold_train_val_df[label_col].nunique() > 1 else None,
        )
        
        # Create data module for this fold
        fold_data_module = KFoldDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tokenizer=tokenizer,
            collator=collator,
            data_path=data_path,
            text_col=text_col,
            label_col=label_col,
            label_names=label_names,
            num_labels=num_labels,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers
        )
        
        yield fold_idx, fold_data_module


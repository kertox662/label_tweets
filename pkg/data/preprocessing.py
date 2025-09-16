'''
Text preprocessing utilities for tweet classification.
This file contains functions to clean and preprocess tweet text data.
'''

import re
from ftfy import fix_text
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers import InputExample
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


def clean_text(text):
    """
    Clean tweet text by removing URLs, processing hashtags and mentions,
    fixing encoding issues, and normalizing whitespace.
    
    Args:
        text (str): Raw tweet text
        
    Returns:
        str: Cleaned tweet text
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'#(\w+)', lambda m: ' '.join(re.findall(r'[A-Z][a-z]*|[a-z]+|\d+', m.group(1))),
                  text)  # Convert hashtags like #HelloWorld to Hello World

    # Error for #'s with shortcuts, i.e. #UTC is changed into `U T C`
    text = re.sub(r'@(\w+)', r'\1', text)  # Remove @ symbols from mentions
    text = fix_text(text)  # Fix encoding issues
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    # Remove extra spaces that may result from word removal
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extra_preprocessing(text):
    """
    Apply additional preprocessing steps to clean text.
    
    Args:
        text (str): Text that has already been cleaned by clean_text()
        
    Returns:
        str: Further processed text
    """
    # Remove invisible Unicode characters
    text = re.sub(r'[\u200E\u200F\u202A-\u202E\u2066\u2069\u200B-\u200D\u00A0\u2000-\u200A\u00AD\uFFFD\u2063]', "", text)
    text = re.sub(r'\brt\b', "retweet", text, flags=re.IGNORECASE)  # Convert rt to retweet
    text = re.sub(r'\bw\b', "with", text, flags=re.IGNORECASE)  # Convert w to with
    return text


def preprocess_text(tweets, text_col="text"):
    """
    Preprocess a DataFrame of tweets by cleaning the text column.
    
    Args:
        tweets (pd.DataFrame): DataFrame containing tweet data
        text_col (str): Name of the column containing tweet text
        
    Returns:
        pd.DataFrame: DataFrame with added 'clean_text' column
    """
    tweets["clean_text"] = tweets[text_col].astype(str).apply(clean_text).apply(extra_preprocessing)
    return tweets


def get_dataloader_unlabeled(df, text_column, target=None, shuffle=True, batch_size=64, drop_last=True):
    """
    Create a DataLoader for unlabeled data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        text_column (str): Name of the text column
        target (str, optional): Name of the target column
        shuffle (bool): Whether to shuffle the data
        batch_size (int): Batch size for the DataLoader
        drop_last (bool): Whether to drop the last incomplete batch
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    if target is None:
        data = df[text_column].tolist()
    else:
        data = list(zip(df[text_column].tolist(), df[target].tolist()))
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size,
                          collate_fn=lambda x: x, drop_last=drop_last)
    return dataloader


def get_dataloader_ST(df, text_column, shuffle=True, batch_size=64, drop_last=True):
    """
    Create a DataLoader for Sentence Transformers with InputExample format.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        text_column (str): Name of the text column
        shuffle (bool): Whether to shuffle the data
        batch_size (int): Batch size for the DataLoader
        drop_last (bool): Whether to drop the last incomplete batch
        
    Returns:
        DataLoader: PyTorch DataLoader with InputExample format
    """
    data = [InputExample(texts=[s, s]) for s in df[text_column].dropna().tolist()]  # [s,s] to allow for siamese pairings
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    return dataloader


def get_dataloader_labeled(df=None, text_column='clean_tweet', id_col='tweet_id', target=None, shuffle=True, batch_size=64, drop_last=True):
    """
    Create a DataLoader for labeled data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        text_column (str): Name of the text column
        id_col (str): Name of the ID column
        target (str): Name of the target column
        shuffle (bool): Whether to shuffle the data
        batch_size (int): Batch size for the DataLoader
        drop_last (bool): Whether to drop the last incomplete batch
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    data = list(zip(df[id_col].tolist(), df[text_column].tolist(), df[target].tolist()))
    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size,
                          collate_fn=lambda x: x, drop_last=drop_last)
    return dataloader


class TweetsDataModuleLegacy(pl.LightningDataModule):
    """
    Legacy DataModule class from the original preprocessing code.
    This is kept for reference but the new TweetsDataModule in data_module.py is preferred.
    """
    def __init__(self, data: pd.DataFrame, batch_size: int = None, target_col: str = 'AR'):
        super().__init__()
        self.data = pd.DataFrame
        self.batch_size = batch_size
        self.target_col = target_col

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.data = preprocess_text(self.data, text_col="text")
        self.data = self.data[['tweet_id', 'clean_text', 'AR', 'MB']]

        self.train, self.test = train_test_split(self.data[['clean_text', 'AR']], train_size=0.8, random_state=2025, shuffle=True)
        self.train, self.val = train_test_split(self.train, train_size=0.8, random_state=2025, shuffle=True)

    def train_dataloader(self):
        return get_dataloader_labeled(self.train, text_column='clean_text', target=self.target_col, shuffle=True, batch_size=32)

    def val_dataloader(self):
        return get_dataloader_labeled(self.val, text_column='clean_text', target=self.target_col, shuffle=True, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, text_column='clean_text', target=self.target_col, shuffle=False, batch_size=32, drop_last=False)

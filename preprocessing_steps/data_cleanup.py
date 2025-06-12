'''
This is the file used to clean the data and translate it into clean readable data for models
If any model needs to have different data simply create a function that has the model name and 
a reason on why it needs data transformed differently
'''

import re
from ftfy import fix_text
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers import InputExample
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'#(\w+)', lambda m: ' '.join(re.findall(r'[A-Z][a-z]*|[a-z]+|\d+', m.group(1))),
                  text)

    # Error for #'s with shortcuts, i.e. #UTC is changed into `U T C`
    text = re.sub(r'@(\w+)', r'\1', text)
    text = fix_text(text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove extra spaces that may result from word removal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extra_preprocessing(text):
    text = re.sub(r'[\u200E\u200F\u202A-\u202E\u2066\u2069\u200B-\u200D\u00A0\u2000-\u200A\u00AD\uFFFD\u2063]',"",text)
    text = re.sub(r'\brt\b', "retweet", text, flags=re.IGNORECASE)
    text = re.sub(r'\bw\b', "with", text, flags=re.IGNORECASE)
    return text


def preprocess_text(tweets, text_col="text"):
    tweets["clean_text"] = tweets[text_col].astype(str).apply(clean_text).apply(extra_preprocessing)
    return tweets


def get_dataloader_unlabeled(df, text_column, target=None, shuffle=True, batch_size=64, drop_last=True):
  if target is None:
    data = df[text_column].tolist()
  else:
    data = list(zip(df[text_column].tolist(), df[target].tolist()))
  dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size,
                          collate_fn=lambda x:x, drop_last=drop_last)
  return dataloader




def get_dataloader_ST(df, text_column, shuffle=True, batch_size=64, drop_last=True):
  data = [InputExample(texts=[s, s]) for s in df[text_column].dropna().tolist()] # [s,s] to allow for siamese pairings.
  dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
  return dataloader


def get_dataloader_labeled(df=None, text_column='clean_tweet', id_col='tweet_id', target=None, shuffle=True, batch_size=64, drop_last=True):
  data = list(zip(df[id_col].tolist(), df[text_column].tolist(), df[target].tolist()))
  dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size,
                          collate_fn=lambda x:x, drop_last=drop_last)
  return dataloader


class tweetsDataModule(pl.LightningDataModule):
  def __init__(self, data: pd.DataFrame, batch_size: int = None, target_col: str = 'AR'):
      super().__init__()
      self.data = pd.DataFrame
      self.batch_size = batch_size
      self.target_col = target_col

  def setup(self, stage: str):
      # Assign train/val datasets for use in dataloaders
      self.data = preprocess_text(self.data, text_col="text")
      self.data = self.data[['tweet_id', 'clean_text', 'AR','MB']]

      self.train, self.test = train_test_split(self.data[['clean_text', 'AR']], train_size=0.8, random_state=2025, shuffle=True)
      self.train, self.val = train_test_split(self.train, train_size=0.8, random_state=2025, shuffle=True)

  def train_dataloader(self):
      return get_dataloader_labeled(self.train, text_column='clean_text', target=self.target_col, shuffle=True, batch_size=32)

  def val_dataloader(self):
      return get_dataloader_labeled(self.val,  text_column='clean_text', target=self.target_col, shuffle=True, batch_size=32)

  def test_dataloader(self):
      return DataLoader(self.test, text_column='clean_text', target=self.target_col, shuffle=False,  batch_size=32, drop_last=False)
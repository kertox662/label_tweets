'''
This is the file used to clean the data and translate it into clean readable data for models
If any model needs to have different data simply create a function that has the model name and 
a reason on why it needs data transformed differently
'''

import re
from ftfy import fix_text
from torch.utils.data import DataLoader

from sentence_transformers import InputExample

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


def preprocess_text(tweets, text_col="text", preprocessed_col="clean_text"):
    tweets[preprocessed_col] = tweets[text_col].astype(str).apply(clean_text).apply(extra_preprocessing)
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

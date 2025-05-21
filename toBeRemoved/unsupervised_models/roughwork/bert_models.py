import os
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, losses, models
import pytorch_lightning as pl

import torch
import torch.nn as nn
import time

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
# Define the neural network
class BERTweetClassifier(nn.Module):
    def __init__(self, bertweet_model_name="vinai/bertweet-base", num_labels=3, device=None):
        super(BERTweetClassifier, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bertweet = AutoModel.from_pretrained(bertweet_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bertweet.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
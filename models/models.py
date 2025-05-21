import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import get_linear_schedule_with_warmup
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



def train_model(model, dataloader, valid_loader, num_epochs, file_path, optimizer, scheduler, criterion):
    print("Initializing", flush=True)
    start_time = time.monotonic()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print(f"Model is on device: {device}", flush=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} ------------------------------------------------", flush=True)

        model.train()
        total_loss = 0

        for i, batch in enumerate(dataloader):
            # print(f"Current batch {i}", flush=True)
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        valid_loss = 0
        all_labels = []
        all_preds = []

        print("\nValidation ------------------------------------------------", flush=True)
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                pred_labels = torch.argmax(probs, dim=1)

                all_labels.append(labels.cpu())
                all_preds.append(pred_labels.cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} Results:", flush=True)
        print(f"  Training Loss: {total_loss / len(dataloader):.4f}", flush=True)
        print(f"  Validation Loss: {valid_loss / len(valid_loader):.4f}", flush=True)
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}", flush=True)

        # implement saving the best model based on f1 / validation loss
    print("Training complete.", flush=True)
    print(f"Total time taken: {round((time.monotonic() - start_time) / 60, 2)} minutes", flush=True)
    return model, all_preds

torch.manual_seed(2025)
bertweet_model_name = "vinai/bertweet-base"
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 5
if train_loader is None or valid_loader is None:
    raise ValueError("Error: train_loader or valid_loader is None")

total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

print("Starting training...", flush=True)

model = BERTweetClassifier(bertweet_model_name=bertweet_model_name, num_labels=3)
# model, probs = train_model(
#     model, train_loader, valid_loader, num_epochs=num_epochs, file_path="",
#     optimizer=optimizer, scheduler=scheduler, criterion=criterion
# )


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

# class SimCSETrainer(pl.LightningModule):
#     def __init__(self,
#                  model_name: str = "bert-base-uncased",
#                  max_seq_length: int = 32,
#                  learning_rate: float = 5e-5,
#                  output_prefix: str = ""):
#         super().__init__()
#         self.model_name = model_name
#         self.max_seq_length = max_seq_length
#         self.learning_rate = learning_rate

#         # Create an output path similar to your original class,
#         # using a timestamp and optional prefix.
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         self.model_output_path = f"output_model/train_simcse{output_prefix}-{timestamp}"

#         # Build the SentenceTransformer model
#         self.model = self._build_model()
#         # Create the loss module. This loss takes care of computing
#         # the contrastive (multiple negatives ranking) loss.
#         self.loss_fn = losses.MultipleNegativesRankingLoss(self.model)

#     def _build_model(self):
#         # Build the model using a Transformer for word embeddings
#         # and a pooling layer to get sentence embeddings.
#         word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
#         pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#         return SentenceTransformer(modules=[word_embedding_model, pooling_model])

#     def forward(self, features):
#         # For inference purposes, pass input features to the SentenceTransformer.
#         # The 'features' input should be formatted as required by your tokenizer/model.
#         return self.model(features)

#     def training_step(self, batch, batch_idx):
#         # Compute loss using the provided loss function.
#         # It is assumed that your train_dataloader yields batches in the appropriate format.
#         loss = self.loss_fn(batch)
#         self.log("train_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         # Create an optimizer. You might later add a scheduler if needed.
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer

#     # Optional: Utility methods to save and load the SentenceTransformer model.
#     def save_model(self, output_path: str = None):
#         path = output_path or self.model_output_path
#         self.model.save(path)
#         print(f"Model saved to: {path}")
#         return path

#     def load_model(self, model_path: str):
#         self.model = SentenceTransformer(model_path)
#         print(f"Model loaded from: {model_path}")


class SimCSETrainer:
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 max_seq_length: int = 32,
                 batch_size: int = 32,
                 num_epochs: int = 1,
                 learning_rate: float = 5e-5,
                 output_prefix: str = ""):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_prefix = output_prefix

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_output_path = f"output_model/train_simcse{output_prefix}-{timestamp}"
        self.model = self._build_model()

    def _build_model(self):
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def train(self, train_dataloader):
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        warmup_steps = int(np.ceil(len(train_dataloader) * self.num_epochs * 0.1))
        print(f"Warmup-steps: {warmup_steps}")

        # Fit the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": self.learning_rate},
            checkpoint_path=self.model_output_path,
            show_progress_bar=True,
            use_amp=False,
        )

    def save_model(self, output_path: str = None):
        path = output_path or self.model_output_path
        self.model.save(path)
        print(f"Model saved to: {path}")
        return path

    def load_model(self, model_path: str):
        self.model = SentenceTransformer(model_path)
        print(f"Model loaded from: {model_path}")



class TweetClassifierHead(nn.Module):
    def __init__(self, embedding_dim=768, num_labels=3, dropout_rate=0.1):
        super(TweetClassifierHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embedding_dim, num_labels)

    def forward(self, embeddings):
        x = self.dropout(embeddings)
        logits = self.fc(x)
        return logits


class DownstreamTrainer:
    def __init__(self, model, train_loader, valid_loader, num_labels=3, lr=5e-5, num_epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.downstream_model = TweetClassifierHead()

        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

    def train(self):
        print("Initializing", flush=True)
        start_time = time.monotonic()
        self.model.to(self.device)
        self.downstream_model.to(self.device)
        print(f"Models is on device: {self.device}", flush=True)

        best_f1 = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs} ------------------------------------------------", flush=True)

            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                texts, labels = zip(*batch)
                labels = torch.tensor(labels).to(self.device)
                self.optimizer.zero_grad()
                encoding = self.model.encode(texts,device=self.device)
                encoding = torch.tensor(encoding).to(self.device)
                outputs = self.downstream_model(encoding)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            print(f"  Training Loss: {total_loss / len(self.train_loader):.4f}", flush=True)

            # Validation
            val_loss, val_dict = self.evaluate()

            # Get F1 from the classification report if needed
            f1 = val_dict['macro avg']['f1-score']
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = self.model.state_dict()

        print("Training complete.", flush=True)
        print(f"Total time taken: {round((time.monotonic() - start_time) / 60, 2)} minutes", flush=True)

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.model

    def evaluate(self):
        valid_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in self.valid_loader:
                texts, labels = zip(*batch)
                labels = torch.tensor(labels).to(self.device)
                self.optimizer.zero_grad()
                encoding = self.model.encode(texts,device=self.device)
                encoding = torch.tensor(encoding).to(self.device)
                outputs = self.downstream_model(encoding)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                pred_labels = torch.argmax(probs, dim=1)

                all_labels.append(labels.cpu())
                all_preds.append(pred_labels.cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        avg_loss = valid_loss / len(self.valid_loader)
        val_dict = classification_report(all_labels, all_preds, digits=4, output_dict=True)
        print(f"  Validation Loss: {avg_loss:.4f}", flush=True)
        print(classification_report(all_labels, all_preds, digits=4))


        return avg_loss, val_dict

    def predict(self, dataloader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                texts, labels = zip(*batch)
                labels = torch.tensor(labels).to(self.device)
                encoding = self.model.encode(texts,device=self.device)
                encoding = torch.tensor(encoding).to(self.device)
                outputs = self.downstream_model(encoding)
                probs = torch.softmax(outputs, dim=1)
                pred_labels = torch.argmax(probs, dim=1)
                all_preds.append(pred_labels.cpu())
        return torch.cat(all_preds).numpy()


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

print("Importing python packages")
import pandas as pd
from sentence_transformers import SentenceTransformer
from data_preprocess import preprocess_text, get_dataloader_unlabeled, get_dataloader_ST
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from models import SimCSETrainer, DownstreamTrainer

# --------------------------------------- Import and pre-process data ---------------------------------------
print("\nImporting data\n")
## Unlabeled
labeled_path = './data/MichelleCoding1500 corrected march_13_2025.xlsx'
tweets = pd.read_excel(labeled_path)
tweets = preprocess_text(tweets, text_col="text")
tweets['AR'] = tweets['AR'] - 1

# path = "./unlabeled_sample_1_percent.csv"
# unlabeled_path = "./data/Unlabeled_full.xlsx"
unlabeled_path = "./data/unlabeled_200.csv"
tweets_unlabeled = pd.read_csv(unlabeled_path).sample(n=65)
# tweets_unlabeled = pd.read_excel(unlabeled_path)
tweets_unlabeled = preprocess_text(tweets_unlabeled, text_col="text")

tweets_labeled = tweets[['id', 'clean_text', 'AR','MB']]

train_set, test_set = train_test_split(tweets_labeled[['clean_text', 'AR']],
                                             train_size=0.8, random_state=2025, shuffle=True)

train_set, validation_set = train_test_split(train_set, train_size=0.8, random_state=2025, shuffle=True)

print(f"train_set: {train_set.shape}")
print(f"validation_set: {validation_set.shape}")
print(f"test_set: {test_set.shape}")

# ------------------------------------------ Prepare data loader ------------------------------------------
print("\nPreparing dataloader\n")
# Unlabeled dataset
train_unlabeled_loader = get_dataloader_ST(tweets_unlabeled, text_column="clean_text", shuffle=True)

# Labeled dataset
target_cols = 'AR'
train_loader = get_dataloader_unlabeled(train_set, text_column='clean_text',
                                        target=target_cols, shuffle=True)
valid_loader = get_dataloader_unlabeled(validation_set, text_column='clean_text',
                                        target=target_cols, shuffle=True)
test_loader = get_dataloader_unlabeled(test_set, text_column='clean_text',
                                        target=target_cols, shuffle=False, drop_last=False)

# --------------------------------------- Train unsupervised model ---------------------------------------
print("\ntraining unsupervised model\n")
model_name = "vinai/bertweet-base"
model_name2 = "digio/Twitter4SSE"

trainer = SimCSETrainer(model_name=model_name2, num_epochs=1, output_prefix="_full_1e")
trainer.train(train_unlabeled_loader)

# model = SimCSETrainer(model_name="bert-base-uncased", max_seq_length=32, learning_rate=5e-5)


# trainer = Trainer(max_epochs=1)  # adjust gpus if needed
# trainer.fit(model, train_unlabeled_loader)
# Save the model (optional, already saved during training)
# path = trainer.save_model()

# Download the zipped model directory
# shutil.make_archive(path, 'zip', path)
# files.download(f'{path}.zip')

# --------------------------------------- Train downstream model ---------------------------------------
print("\ntraining downstream model\n")
model_path = "output_model/train_simcse-twitter-2025-03-25_18-05-12"
# simcse_model = SentenceTransformer(model_path)
simcse_model = trainer.model
downstream_trainer = DownstreamTrainer(simcse_model, train_loader, valid_loader, lr=5e-5, num_epochs=1)
downstream_model = downstream_trainer.train()

# ----------------------------------------- Evaluate test data -----------------------------------------
print("\nPredicting test data\n")
test_set['predicted_label'] = downstream_trainer.predict(test_loader)
print(classification_report(test_set[["AR"]], test_set[["predicted_label"]]))
# Pseudo-Labeling Training Scripts

## Setup

1. Create virtual environment and install packages:
```bash
python3 -m venv label_tweets/venv
source label_tweets/venv/bin/activate
pip install -r label_tweets/requirements.txt
```

## Pseudo-Training (`pseudo_train.py`)

Iteratively trains a model on labeled data, then uses high-confidence predictions to pseudo-label unlabeled data.

**Note:** Prepare your CSV files with the following columns:
- `labelled_csv`: Must have `text`, `AR` (label), and `tweet_id` columns
- `unlabelled_csv`: Must have `text` and `tweet_id` columns (no `AR` column)
- `test_csv`: Must have `text`, `AR` (label), and `tweet_id` columns

**Important:** The three datasets must have NO overlapping tweets. The intersection of tweet_ids between any pair must be empty. 


**Usage:**
```bash
python pseudo_train.py \
  --labelled_csv tweets_data/LABELLED.csv \
  --unlabelled_csv tweets_data/UNLABELLED.csv \
  --test_csv tweets_data/TEST.csv \
  --batch_size 128 \
  --max_epochs 6 \
  --max_iterations 10 \
  --min_confidence 0.7 \
  --confidence_step 0.05 \
  --class_weight \
  --output_dir pseudo_training_runs
```

**Parameters:**
- `--labelled_csv`: Initial labeled training data
- `--unlabelled_csv`: Unlabeled data to pseudo-label
- `--test_csv`: Test set for evaluation
- `--batch_size`: Training batch size (default: 64)
- `--max_epochs`: Epochs per iteration (default: 6)
- `--max_iterations`: Max pseudo-labeling iterations (default: 10)
- `--min_confidence`: Minimum confidence threshold (default: 0.7)
- `--confidence_step`: Step size for adaptive threshold (default: 0.05)
- `--class_weight`: Use class weights for imbalanced data
- `--output_dir`: Output directory for results

**How it works:**
1. Train model on labeled data
2. Test on gold test set
3. Predict on unlabeled data
4. Select high-confidence predictions as pseudo-labels
5. Add pseudo-labels to training set
6. Repeat until max iterations or no more high-confidence predictions


Tests GPT models via Martian API on your gold test set.

**Usage:**
```bash
python test_martian_gpt.py --model openai/gpt-4.1-nano
```

**Parameters:**
- `--model`: Model name (e.g., `openai/gpt-4.1-nano`)

**How it works:**
1. Loads gold test set
2. Sends each tweet to Martian GPT API for classification
3. Compares predictions to true labels
4. Reports overall and per-class accuracy

**Output:** Saves results to `martian_gpt_results_TIMESTAMP.csv` and summary JSON.

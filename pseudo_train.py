"""
Iterative pseudo-labeling training pipeline with adaptive confidence thresholds.
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import subprocess
import sys

sys.path.insert(0, 'label_tweets')
from modules.berttweet import BertweetModule, MODEL_NAME, DataCollator


class UnlabeledTweetDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(text, truncation=True, padding=False, 
                           max_length=self.max_length, return_tensors=None)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}


def predict_on_unlabeled(model_path, data_csv, text_col, id_col, batch_size, device):
    """Run inference on unlabeled data and return all predictions with confidence"""
    print(f"\n  Running inference on {data_csv}...")
    
    model = BertweetModule.load_from_checkpoint(model_path, strict=False)
    model = model.to(device)
    model.eval()
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, normalization=True)
    collator = DataCollator(tokenizer)
    
    df = pd.read_csv(data_csv)
    texts = df[text_col].astype(str).tolist()
    tweet_ids = df[id_col].tolist()
    
    dataset = UnlabeledTweetDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, collate_fn=collator)
    
    all_preds, all_confidences = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Predicting", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1) + 1
            confidences = probs.max(axis=1)
            all_preds.append(preds)
            all_confidences.append(confidences)
    
    all_preds = np.concatenate(all_preds)
    all_confidences = np.concatenate(all_confidences)
    
    results = []
    for i in range(len(texts)):
        results.append({
            'text': texts[i],
            'tweet_id': tweet_ids[i],
            'pred_label': int(all_preds[i]),
            'confidence': float(all_confidences[i])
        })
    
    return pd.DataFrame(results)


def save_iteration_markdown(run_dir, iteration, metrics, selected_df):
    """Save detailed markdown report for this iteration"""
    md_file = os.path.join(run_dir, f'iteration_{iteration}_report.md')
    
    with open(md_file, 'w') as f:
        f.write(f"# Pseudo-Training Iteration {iteration} Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset sizes
        f.write("## Dataset Sizes\n\n")
        f.write(f"- **Labeled Data**: {metrics['labeled_size']:,} samples\n")
        f.write(f"- **Unlabeled Pool**: {metrics['unlabeled_size']:,} samples\n")
        f.write(f"- **Pseudo-labels Added**: {metrics['added']:,} samples\n")
        f.write(f"- **Confidence Threshold**: {metrics['confidence_threshold']:.3f}\n\n")
        
        # Test performance
        f.write("## Test Performance\n\n")
        f.write(f"- **Accuracy**: {metrics['test_accuracy']:.4f}\n")
        f.write(f"- **Macro F1**: {metrics['test_f1_macro']:.4f}\n")
        f.write(f"- **Weighted F1**: {metrics['test_f1_weighted']:.4f}\n\n")
        
        # Per-class performance
        f.write("## Per-Class Performance\n\n")
        f.write("| Class | F1 | Precision | Recall | Support |\n")
        for i, (f1, prec, rec, sup) in enumerate(zip(
            metrics['test_f1_per_class'],
            metrics['test_precision_per_class'],
            metrics['test_recall_per_class'],
            metrics['test_support_per_class']
        )):
            f.write(f"| Class {i+1} | {f1:.4f} | {prec:.4f} | {rec:.4f} | {sup} |\n")
        f.write("\n")
        
       
        
        # Pseudo-label analysis
        if selected_df is not None and len(selected_df) > 0:
            f.write("## Pseudo-Label Analysis\n\n")
            f.write(f"- **Confidence Range**: [{selected_df['confidence'].min():.4f}, {selected_df['confidence'].max():.4f}]\n")
            f.write(f"- **Mean Confidence**: {selected_df['confidence'].mean():.4f}\n")
            f.write(f"- **Label Distribution**: {dict(selected_df['pred_label'].value_counts().sort_index())}\n\n")
            
            # Class percentages
            label_counts = selected_df['pred_label'].value_counts().sort_index()
            total = len(selected_df)
            percentages = {k: v/total*100 for k, v in label_counts.items()}
            f.write("### Class Distribution in Pseudo-labels\n\n")
            f.write("| Class | Count | Percentage |\n")
            for class_id in sorted(percentages.keys()):
                f.write(f"| Class {class_id} | {label_counts[class_id]:,} | {percentages[class_id]:.1f}% |\n")
            f.write("\n")
        
        # Model info
        f.write("## Model Information\n\n")
        f.write(f"- **Checkpoint**: `{metrics['model_path']}`\n")
        f.write(f"- **Training Data**: `iteration_{iteration}/train_data.csv`\n")
        f.write(f"- **Unlabeled Data**: `iteration_{iteration}/unlabelled_data.csv`\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        if metrics['test_f1_macro'] > 0.6:
            f.write("**Good Performance**: Macro F1 > 0.6\n")
        else:
            f.write("**Low Macro F1**: Consider adjusting confidence thresholds or class balance\n")
        
        if metrics['added'] > 10000:
            f.write("**Large Addition**: Many pseudo-labels added - monitor quality\n")
        elif metrics['added'] < 100:
            f.write("**Small Addition**: Few pseudo-labels - may need lower confidence threshold\n")
        
        f.write(f"\n**Next Steps**: Monitor per-class F1 scores and pseudo-label quality\n")


def test_model(model_path, test_csv, text_col, label_col, batch_size, num_labels, device):
    """Test model on held-out test set and return metrics"""
    print(f"\n  Testing model on {test_csv}...")
    
    model = BertweetModule.load_from_checkpoint(model_path, strict=False)
    model = model.to(device)
    model.eval()
    
    # Explicitly disable dropout in the underlying transformer model
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, normalization=True)
    collator = DataCollator(tokenizer)
    
    df = pd.read_csv(test_csv)
    texts = df[text_col].astype(str).tolist()
    true_labels = df[label_col].astype(int).tolist()
    true_labels = [l - 1 for l in true_labels]  # Convert to 0-indexed
    
    dataset = UnlabeledTweetDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, collate_fn=collator)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Testing", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.append(preds)
            all_probs.append(probs)
    
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(true_labels, all_preds)
    f1_macro = f1_score(true_labels, all_preds, average='macro')
    f1_weighted = f1_score(true_labels, all_preds, average='weighted')
    f1_per_class = f1_score(true_labels, all_preds, average=None)
    
    # Calculate precision and recall per class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, all_preds, average=None)
    
    # Print detailed metrics
    print(f"\n  DETAILED TEST RESULTS:")
    print(f"    Overall Accuracy: {accuracy:.4f}")
    print(f"    Macro F1: {f1_macro:.4f}")
    print(f"    Weighted F1: {f1_weighted:.4f}")
    print(f"    \n  Per-Class Performance:")
    for i in range(num_labels):
        print(f"    Class {i+1}: F1={f1_per_class[i]:.4f}, Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, Support={support[i]}")
    
    # Print confusion matrix
    cm = confusion_matrix(true_labels, all_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:  0    1    2")
    for i, row in enumerate(cm):
        print(f"    Actual {i}: {row}")
    
    # Print classification report
    print(f"\n  Classification Report:")
    print(classification_report(true_labels, all_preds, target_names=[f'Class {i+1}' for i in range(num_labels)]))
    
    # Calculate class distribution in predictions vs true labels
    from collections import Counter
    true_dist = Counter(true_labels)
    pred_dist = Counter(all_preds)
    print(f"\n  Class Distribution:")
    print(f"    True labels:  {dict(true_dist)}")
    print(f"    Predictions:  {dict(pred_dist)}")
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': f1_per_class.tolist(),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'support_per_class': support.tolist(),
        'num_samples': len(true_labels),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def train_model(train_csv, output_dir, iteration, args):
    """Train model from scratch on labeled data"""
    print(f"ITERATION {iteration}: TRAINING")
    print(f"  Training data: {train_csv}")
    print(f"  Training samples: {len(pd.read_csv(train_csv))}")
    
    # Build training command
    cmd = [
        'python', 'label_tweets/train.py',
        '--data_csv', train_csv,
        '--num_labels', str(args.num_labels),
        '--text_col', args.text_col,
        '--label_col', args.label_col,
        '--val_size', str(args.val_size),
        '--test_size', '0.0',  # No internal test split - use external GOLD_TEST.csv only
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--max_epochs', str(args.max_epochs),
        '--output_dir', output_dir,
    ]
    
    if args.class_weight:
        cmd.append('--class_weight')
    
    print(f"\n  Running command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nTraining failed!")
        return None
    
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and len(d) == 13]
    
    if not subdirs:
        print(f"No timestamped output directory found in {output_dir}")
        return None
    
    # Use the most recent one
    latest_subdir = sorted(subdirs)[-1]
    ckpt_dir = os.path.join(output_dir, latest_subdir, 'ckpt')
    
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return None
    
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and 'last' not in f]
    if not ckpts:
        print(f"No checkpoint found in {ckpt_dir}")
        return None
    
    best_ckpt = os.path.join(ckpt_dir, ckpts[0])
    print(f"\n  Training complete! Checkpoint: {best_ckpt}")
    
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description='Iterative pseudo-labeling with adaptive confidence')
    
    # Data paths
    parser.add_argument('--labelled_csv', type=str, required=True
    parser.add_argument('--unlabelled_csv', type=str, required=True
    parser.add_argument('--test_csv', type=str, required=True
    parser.add_argument('--output_dir', type=str, default='pseudo_training_runs'
    
    # Column names
    parser.add_argument('--text_col', type=str, default='text'
    parser.add_argument('--label_col', type=str, default='AR'
    parser.add_argument('--id_col', type=str, default='tweet_id'
    
    # Pseudo-labeling parameters
    parser.add_argument('--min_confidence', type=float, default=0.7
    parser.add_argument('--confidence_step', type=float, default=0.05
    parser.add_argument('--max_iterations', type=int, default=10
    
    # Training parameters
    parser.add_argument('--num_labels', type=int, default=3
    parser.add_argument('--batch_size', type=int, default=64
    parser.add_argument('--lr', type=float, default=2e-5
    parser.add_argument('--max_epochs', type=int, default=6
    parser.add_argument('--val_size', type=float, default=0.15
    parser.add_argument('--test_size', type=float, default=0.0
    parser.add_argument('--class_weight', action='store_true'
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"\nPSEUDO-LABELING ITERATIVE TRAINING")
    print(f"Labeled data: {args.labelled_csv}")
    print(f"Unlabeled pool: {args.unlabelled_csv}")
    print(f"Test set: {args.test_csv}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Confidence step: {args.confidence_step}")
    print(f"Max iterations: {'Unlimited' if args.max_iterations == -1 else args.max_iterations}")
    print(f"Output: {run_dir}")
    print(f"Device: {device}")
    
    # Initialize working datasets
    labelled_df = pd.read_csv(args.labelled_csv)
    unlabelled_df = pd.read_csv(args.unlabelled_csv)
    
    print(f"\nInitial labeled: {len(labelled_df)}")
    print(f"Initial unlabeled: {len(unlabelled_df)}")
    
    # Track metrics across iterations
    all_metrics = []
    
    iteration = 1
    while True:
        if args.max_iterations != -1 and iteration > args.max_iterations:
            print(f"\nReached max iterations ({args.max_iterations}). Stopping.")
            break
        
        # Save current labeled data for this iteration
        iter_dir = os.path.join(run_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        iter_train_csv = os.path.join(iter_dir, "train_data.csv")
        labelled_df.to_csv(iter_train_csv, index=False)
        
        # Step 1: Train model
        model_ckpt = train_model(iter_train_csv, iter_dir, iteration, args)
        
        if model_ckpt is None:
            print(f"\nTraining failed for iteration {iteration}. Stopping.")
            break
        
        # Test model on held-out test set
        test_metrics = test_model(model_ckpt, args.test_csv, args.text_col, 
                                  args.label_col, args.batch_size, args.num_labels, device)
        
        print(f"\n  Test Results:")
        print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    F1 Macro: {test_metrics['f1_macro']:.4f}")
        print(f"    F1 Weighted: {test_metrics['f1_weighted']:.4f}")
        
        # Step 2: Run inference on unlabeled data
        if len(unlabelled_df) == 0:
            print(f"\nNo more unlabeled data. Stopping.")
            break
        
        iter_unlabelled_csv = os.path.join(iter_dir, "unlabelled_data.csv")
        unlabelled_df.to_csv(iter_unlabelled_csv, index=False)
        
        predictions_df = predict_on_unlabeled(model_ckpt, iter_unlabelled_csv, 
                                              args.text_col, args.id_col, 
                                              args.batch_size, device)
        
        # Log confidence statistics
        print(f"\n  Confidence Statistics on Unlabeled Data ({len(predictions_df)} samples):")
        print(f"  {'='*70}")
        confidences = predictions_df['confidence'].values
        print(f"    Lowest:  {confidences.min():.4f}")
        print(f"    Highest: {confidences.max():.4f}")
        print(f"    Mean:    {confidences.mean():.4f}")
        print(f"    Median:  {np.median(confidences):.4f}")
        print(f"    Std Dev: {confidences.std():.4f}")
        
        # Confidence distribution in bins
        print(f"\n  Confidence Distribution:")
        bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.85), 
                (0.85, 0.9), (0.9, 0.95), (0.95, 0.97), (0.97, 0.99), (0.99, 1.0)]
        for low, high in bins:
            count = ((confidences >= low) & (confidences < high)).sum()
            pct = 100 * count / len(confidences)
            bar = '█' * int(pct / 2)  # Scale for visualization
            print(f"    [{low:.2f}, {high:.2f}): {count:7,} ({pct:5.2f}%) {bar}")
        
        # High confidence breakdown (more granular)
        print(f"\n  High Confidence Breakdown (>= 0.9):")
        high_conf_bins = [(0.90, 0.91), (0.91, 0.92), (0.92, 0.93), (0.93, 0.94), 
                          (0.94, 0.95), (0.95, 0.96), (0.96, 0.97), (0.97, 0.98), 
                          (0.98, 0.99), (0.99, 1.0)]
        for low, high in high_conf_bins:
            count = ((confidences >= low) & (confidences < high)).sum()
            print(f"    [{low:.2f}, {high:.2f}): {count:7,}")
        
        # Label distribution in predictions
        print(f"\n  Predicted Label Distribution:")
        label_counts = predictions_df['pred_label'].value_counts().sort_index()
        for label, count in label_counts.items():
            pct = 100 * count / len(predictions_df)
            print(f"    Label {label}: {count:7,} ({pct:5.2f}%)")
        print(f"  {'='*70}\n")
        
        # Step 3: Adaptive confidence selection
        print(f"\n  Adaptive confidence selection:")
        current_threshold = 1.0
        selected_df = None
        
        while current_threshold >= args.min_confidence:
            # Find predictions in range [current_threshold, 1.0]
            candidates = predictions_df[predictions_df['confidence'] >= current_threshold]
            
            if len(candidates) > 0:
                print(f"    Found {len(candidates)} predictions with confidence >= {current_threshold:.3f}")
                
                # Detailed analysis of selected pseudo-labels
                print(f"    Pseudo-label Analysis:")
                print(f"      Confidence range: [{candidates['confidence'].min():.4f}, {candidates['confidence'].max():.4f}]")
                print(f"      Mean confidence: {candidates['confidence'].mean():.4f}")
                print(f"      Label distribution: {dict(candidates['pred_label'].value_counts().sort_index())}")
                
                # Check for class imbalance in pseudo-labels
                label_counts = candidates['pred_label'].value_counts().sort_index()
                total = len(candidates)
                print(f"      Class percentages: {dict((label_counts / total * 100).round(1))}")
                
                selected_df = candidates
                break
            else:
                print(f"    × No predictions with confidence >= {current_threshold:.3f}")
                current_threshold -= args.confidence_step
        
        if selected_df is None or len(selected_df) == 0:
            print(f"\nNo predictions found above minimum confidence {args.min_confidence}. Stopping.")
            
            # Save final metrics
            all_metrics.append({
                'iteration': iteration,
                'labeled_size': len(labelled_df),
                'unlabeled_size': len(unlabelled_df),
                'added': 0,
                'confidence_threshold': current_threshold,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'test_f1_weighted': test_metrics['f1_weighted'],
                'model_path': model_ckpt
            })
            break
        
        # Step 4: Add to labeled, remove from unlabeled
        print(f"\n  Adding {len(selected_df)} pseudo-labels to training set")
        print(f"    Confidence range: [{selected_df['confidence'].min():.4f}, {selected_df['confidence'].max():.4f}]")
        print(f"    Label distribution: {dict(selected_df['pred_label'].value_counts().sort_index())}")
        
        # Create new labeled entries (match columns with original labeled data)
        new_labeled = pd.DataFrame({
            args.text_col: selected_df[args.text_col],
            args.label_col: selected_df['pred_label']
        })
        
        # Add MB column matching AR for pseudo-labels (for compatibility with berttweet.py)
        new_labeled['MB'] = new_labeled[args.label_col]
        
        # Add any other columns that exist in both
        for col in labelled_df.columns:
            if col not in new_labeled.columns and col in selected_df.columns:
                new_labeled[col] = selected_df[col]
        
        # Update datasets
        labelled_df = pd.concat([labelled_df, new_labeled], ignore_index=True)
        
        selected_ids = set(selected_df[args.id_col].tolist())
        unlabelled_df = unlabelled_df[~unlabelled_df[args.id_col].isin(selected_ids)]
        
        print(f"\n  Updated sizes:")
        print(f"    Labeled: {len(labelled_df)} (+{len(selected_df)})")
        print(f"    Unlabeled: {len(unlabelled_df)} (-{len(selected_df)})")
        
        # Save confidence statistics for this iteration
        confidence_stats = {
            'min': float(confidences.min()),
            'max': float(confidences.max()),
            'mean': float(confidences.mean()),
            'median': float(np.median(confidences)),
            'std': float(confidences.std()),
            'prediction_dist': {int(k): int(v) for k, v in predictions_df['pred_label'].value_counts().sort_index().items()}
        }
        
        confidence_file = os.path.join(iter_dir, 'confidence_stats.json')
        with open(confidence_file, 'w') as f:
            json.dump(confidence_stats, f, indent=2)
        
        # Save metrics for this iteration
        iteration_metrics = {
            'iteration': iteration,
            'labeled_size': len(labelled_df),
            'unlabeled_size': len(unlabelled_df),
            'added': len(selected_df),
            'confidence_threshold': current_threshold,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_f1_per_class': test_metrics['f1_per_class'],
            'test_precision_per_class': test_metrics['precision_per_class'],
            'test_recall_per_class': test_metrics['recall_per_class'],
            'test_support_per_class': test_metrics['support_per_class'],
            'test_confusion_matrix': test_metrics['confusion_matrix'],
            'model_path': model_ckpt
        }
        all_metrics.append(iteration_metrics)
        
        # Save metrics to JSON
        metrics_file = os.path.join(run_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save detailed markdown report for this iteration
        save_iteration_markdown(run_dir, iteration, iteration_metrics, selected_df)
        
        iteration += 1
    
    # Final summary
    print(f"\nTRAINING COMPLETE!")
    print(f"Total iterations: {len(all_metrics)}")
    print(f"Final labeled size: {labelled_df.shape[0] if len(all_metrics) > 0 else 'N/A'}")
    print(f"Final unlabeled size: {unlabelled_df.shape[0] if len(all_metrics) > 0 else 'N/A'}")
    
    if all_metrics:
        print(f"\nTest Accuracy Progression:")
        for m in all_metrics:
            print(f"  Iteration {m['iteration']}: {m['test_accuracy']:.4f} (F1: {m['test_f1_macro']:.4f}, +{m['added']} samples)")
        
        print(f"\nBest model: Iteration {max(all_metrics, key=lambda x: x['test_f1_macro'])['iteration']}")
        print(f"Best F1 Macro: {max(m['test_f1_macro'] for m in all_metrics):.4f}")
    
    print(f"\nResults saved to: {run_dir}")
    
    # Save final datasets
    final_labeled_path = os.path.join(run_dir, 'final_labeled_data.csv')
    final_unlabeled_path = os.path.join(run_dir, 'final_unlabeled_data.csv')
    labelled_df.to_csv(final_labeled_path, index=False)
    unlabelled_df.to_csv(final_unlabeled_path, index=False)
    print(f"\nFinal datasets saved:")
    print(f"  Labeled: {final_labeled_path}")
    print(f"  Unlabeled: {final_unlabeled_path}")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse

def classify_with_threshold(df, threshold):
    """
    Classify tweets based on the threshold:
    - If confidence for 3 (implicit) > threshold: classify as 3
    - Otherwise: classify as the higher of 1 or 2
    """
    predictions = []
    
    for idx, row in df.iterrows():
        conf1 = row['class_1_confidence']
        conf2 = row['class_2_confidence']
        true_label = row['AR']
        
        # Skip if confidence scores are missing
        if pd.isna(conf1) or pd.isna(conf2):
            continue
        
        # Calculate confidence for class 3
        conf3 = 100 - conf1 - conf2
        
        # Classify based on threshold
        if conf3 > threshold:
            pred = 3
        else:
            # Choose between class 1 and 2 based on higher confidence
            pred = 1 if conf1 >= conf2 else 2
        
        predictions.append({
            'true_label': true_label,
            'predicted_label': pred,
            'conf1': conf1,
            'conf2': conf2,
            'conf3': conf3
        })
    
    return pd.DataFrame(predictions)

def evaluate_threshold(df, threshold):
    """
    Evaluate a threshold and return accuracy metrics
    """
    results_df = classify_with_threshold(df, threshold)
    
    if len(results_df) == 0:
        return None
    
    accuracy = accuracy_score(results_df['true_label'], results_df['predicted_label'])
    f1_macro = f1_score(results_df['true_label'], results_df['predicted_label'], average='macro')
    f1_weighted = f1_score(results_df['true_label'], results_df['predicted_label'], average='weighted')
    
    # Per-class F1
    f1_per_class = f1_score(results_df['true_label'], results_df['predicted_label'], average=None)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_class1': f1_per_class[0],
        'f1_class2': f1_per_class[1],
        'f1_class3': f1_per_class[2],
        'num_samples': len(results_df)
    }

def main():
    parser = argparse.ArgumentParser(description='Optimize classification threshold for implicit class 3')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--start_threshold', type=float, default=1.0)
    parser.add_argument('--end_threshold', type=float, default=100.0)
    parser.add_argument('--step', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load the data
    df = pd.read_csv(args.input_csv)
    
    print(f"Loaded {len(df)} rows from {args.input_csv}")
    print(f"Testing thresholds from {args.start_threshold} to {args.end_threshold} with step {args.step}")
    print()
    
    # Test different thresholds
    results = []
    for threshold in np.arange(args.start_threshold, args.end_threshold + args.step, args.step):
        metrics = evaluate_threshold(df, threshold)
        if metrics:
            results.append(metrics)
            print(f"Threshold {threshold:5.1f}: Accuracy={metrics['accuracy']:.4f}, F1 Macro={metrics['f1_macro']:.4f}, F1 Weighted={metrics['f1_weighted']:.4f}")
    
    if not results:
        print("No valid results found")
        return
    
    # Find best threshold
    results_df = pd.DataFrame(results)
    
    # Best by accuracy
    best_acc = results_df.loc[results_df['accuracy'].idxmax()]
    print()
    print("=" * 80)
    print("BEST THRESHOLD BY ACCURACY:")
    print("=" * 80)
    print(f"Threshold: {best_acc['threshold']:.1f}")
    print(f"Accuracy: {best_acc['accuracy']:.4f} ({best_acc['accuracy']*100:.2f}%)")
    print(f"F1 Macro: {best_acc['f1_macro']:.4f}")
    print(f"F1 Weighted: {best_acc['f1_weighted']:.4f}")
    print(f"F1 Class 1: {best_acc['f1_class1']:.4f}")
    print(f"F1 Class 2: {best_acc['f1_class2']:.4f}")
    print(f"F1 Class 3: {best_acc['f1_class3']:.4f}")
    print(f"Samples: {best_acc['num_samples']}")
    print()
    
    # Best by F1 macro
    best_f1 = results_df.loc[results_df['f1_macro'].idxmax()]
    print("=" * 80)
    print("BEST THRESHOLD BY F1 MACRO:")
    print("=" * 80)
    print(f"Threshold: {best_f1['threshold']:.1f}")
    print(f"Accuracy: {best_f1['accuracy']:.4f} ({best_f1['accuracy']*100:.2f}%)")
    print(f"F1 Macro: {best_f1['f1_macro']:.4f}")
    print(f"F1 Weighted: {best_f1['f1_weighted']:.4f}")
    print(f"F1 Class 1: {best_f1['f1_class1']:.4f}")
    print(f"F1 Class 2: {best_f1['f1_class2']:.4f}")
    print(f"F1 Class 3: {best_f1['f1_class3']:.4f}")
    print(f"Samples: {best_f1['num_samples']}")
    print()
    
    # Save all results
    output_file = args.input_csv.replace('.csv', '_threshold_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"All results saved to {output_file}")

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import ast
import argparse
from pathlib import Path

def calculate_f1_scores_from_cm(confusion_matrix):
    """
    Calculate macro and weighted F1 scores from a confusion matrix.
    
    Args:
        confusion_matrix: 2D array/list representing the confusion matrix
        
    Returns:
        tuple: (macro_f1, weighted_f1, per_class_f1)
    """
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]
    
    f1_scores = []
    precisions = []
    recalls = []
    supports = []
    
    for i in range(n_classes):
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: sum of column i excluding diagonal
        fp = cm[:, i].sum() - tp
        
        # False Negatives: sum of row i excluding diagonal
        fn = cm[i, :].sum() - tp
        
        # Support: actual number of samples in this class
        support = cm[i, :].sum()
        supports.append(support)
        
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # Macro F1: unweighted average
    macro_f1 = np.mean(f1_scores)
    
    # Weighted F1: weighted by support
    total_support = sum(supports)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1_scores, supports)) / total_support if total_support > 0 else 0
    
    return macro_f1, weighted_f1, f1_scores

def process_csv_file(input_file, output_file=None, transpose=False):
    """
    Read a CSV file with confusion matrices and recalculate F1 scores.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input)
        transpose: If True, transpose confusion matrices before calculating
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Processing {input_file}")
    print(f"Found {len(df)} rows")
    
    # Check if required columns exist
    if 'cm' not in df.columns:
        print("Error: 'cm' column not found in CSV")
        return
    
    # Process each row
    new_macro_f1 = []
    new_weighted_f1 = []
    
    for idx, row in df.iterrows():
        # Parse the confusion matrix string
        cm_str = row['cm']
        try:
            cm = ast.literal_eval(cm_str)
            
            # Optionally transpose
            if transpose:
                cm = np.array(cm).T.tolist()
            
            # Calculate F1 scores
            macro_f1, weighted_f1, per_class_f1 = calculate_f1_scores_from_cm(cm)
            
            new_macro_f1.append(macro_f1)
            new_weighted_f1.append(weighted_f1)
            
            if idx < 3:  # Print first few for verification
                print(f"\nRow {idx}:")
                print(f"  CM: {cm}")
                print(f"  Per-class F1: {[f'{f:.4f}' for f in per_class_f1]}")
                old_macro_col = 'test_macro_f1' if 'test_macro_f1' in df.columns else 'test_macro_F1'
                old_weighted_col = 'test_weighted_f1' if 'test_weighted_f1' in df.columns else 'test_weighted_F1'
                if old_macro_col in df.columns and pd.notna(row.get(old_macro_col)):
                    print(f"  Old Macro F1: {row[old_macro_col]:.4f}")
                else:
                    print(f"  Old Macro F1: N/A")
                print(f"  New Macro F1: {macro_f1:.4f}")
                if old_weighted_col in df.columns and pd.notna(row.get(old_weighted_col)):
                    print(f"  Old Weighted F1: {row[old_weighted_col]:.4f}")
                else:
                    print(f"  Old Weighted F1: N/A")
                print(f"  New Weighted F1: {weighted_f1:.4f}")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Keep old values if parsing fails
            new_macro_f1.append(row.get('test_macro_f1', np.nan))
            new_weighted_f1.append(row.get('test_weighted_f1', np.nan))
    
    # Update the dataframe
    df['test_macro_f1'] = new_macro_f1
    df['test_weighted_f1'] = new_weighted_f1
    
    # Save to output file
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"\nUpdated CSV saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recalculate F1 scores from confusion matrices in CSV files')
    parser.add_argument('input_file', type=str, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output CSV file path (default: overwrite input)')
    parser.add_argument('--transpose', '-t', action='store_true', help='Transpose confusion matrices before calculating')
    
    args = parser.parse_args()
    
    process_csv_file(args.input_file, args.output, args.transpose)
    

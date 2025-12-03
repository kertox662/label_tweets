#!/usr/bin/env python3
import pandas as pd
import argparse

def count_labels(csv_file, label_col='AR'):
    df = pd.read_csv(csv_file)
    
    print(f"Dataset: {csv_file}")
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(f"{'Label':<10} {'Count':<10} {'Percentage':<15}")
    print("-" * 35)
    
    counts = df[label_col].value_counts().sort_index()
    
    for label in sorted(counts.index):
        count = counts[label]
        percentage = (count / len(df)) * 100
        print(f"{label:<10} {count:<10} {percentage:>6.2f}%")
    
    print("-" * 35)
    print(f"{'Total':<10} {len(df):<10} {100.00:>6.2f}%")
    
    return counts

def main():
    parser = argparse.ArgumentParser(description='Count label distribution in dataset')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--label_col', type=str, default='AR')
    
    args = parser.parse_args()
    
    counts = count_labels(args.input_csv, args.label_col)

if __name__ == '__main__':
    main()

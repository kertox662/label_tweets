#!/usr/bin/env python3
import pandas as pd
import argparse

def remove_overlap(source_file, target_file, output_file, id_col='tweet_id'):
    df1 = pd.read_csv(source_file)
    df2 = pd.read_csv(target_file)
    overlap = df1[id_col].isin(df2[id_col])
    df2_filtered = df2[~df2[id_col].isin(df1[id_col])]
    df2_filtered.to_csv(output_file, index=False)
    print(f"Removed {overlap.sum()} overlapping tweets from {target_file}")
    print(f"Saved {len(df2_filtered)} tweets to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove overlapping tweets between two files')
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()
    
    remove_overlap(args.source, args.target, args.output)

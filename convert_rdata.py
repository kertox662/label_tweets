#!/usr/bin/env python3
import pyreadr
import argparse
import sys

def convert_rdata(input_file, output_file):
    try:
        result = pyreadr.read_r(input_file)
        if len(result) == 0:
            print(f"Error: No data found in {input_file}")
            sys.exit(1)
        df = result[list(result.keys())[0]]
        df.to_csv(output_file, index=False)
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Rdata file to CSV')
    parser.add_argument('input', type=str)
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()
    
    output_file = args.output
    if output_file is None:
        output_file = args.input.replace('.Rdata', '.csv')
    
    convert_rdata(args.input, output_file)

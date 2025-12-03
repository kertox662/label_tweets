#!/usr/bin/env python3
import pandas as pd
import requests
import time
import argparse
import json
from datetime import datetime

MARTIAN_URL = "https://api.withmartian.com/v1/chat/completions"
MARTIAN_API_KEY = ""

def classify_tweet_with_confidence(text, model_name, api_key=MARTIAN_API_KEY):
    """
    Classify a single tweet and get confidence scores for classes 1 and 2
    """
    
    prompt = f"""Based on Kingdon's theory, please provide confidence scores for this tweet.

Classify this tweet into these categories:
1. Problem Oriented - The tweet describes or mentions a problem, issue, or challenge
2. Solution Oriented - The tweet describes or mentions a solution, fix, or resolution  

For this tweet, please provide:
- Confidence score for Class 1 (Problem) as a percentage (0-100)
- Confidence score for Class 2 (Solution) as a percentage (0-100)

Note: The sum of both confidence scores should not exceed 100%, may leave room for other.

Format your response as: CONF1,CONF2 [explanation]
For example: 85,10 Political discussion with some problem elements

Tweet: {text}"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 150
    }

    try:
        response = requests.post(MARTIAN_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        prediction_text = result['choices'][0]['message']['content'].strip()

        # Parse the response format: CONF1,CONF2
        try:
            parts = prediction_text.split(',')
            if len(parts) >= 2:
                conf1 = float(parts[0].strip())
                # Extract just the number from the second part (ignore explanation)
                conf2_str = parts[1].strip()
                conf2 = float(conf2_str.split()[0])
                
                # Extract explanation (everything after the first two numbers)
                explanation = prediction_text.split(',', 1)[1].strip()
                if len(explanation.split()) > 1:
                    explanation = ' '.join(explanation.split()[1:])
                else:
                    explanation = ''
                
                # Validate ranges and ensure total doesn't exceed 100%
                if 0 <= conf1 <= 100 and 0 <= conf2 <= 100 and (conf1 + conf2) <= 100:
                    return conf1, conf2, explanation
        except Exception as parse_err:
            pass
        
        return None, None, None

    except Exception as e:
        print(f"API Error: {e}")
        return None, None, None

def test_martian_confidence_on_csv(model_name, input_csv, output_csv):
    # Load the input data
    df = pd.read_csv(input_csv)
    
    # Initialize results
    conf1_scores = []
    conf2_scores = []
    valid_indices = []
    total_rows = len(df)
    
    # Process each tweet
    for idx, row in df.iterrows():
        text = row['text']
        
        result = classify_tweet_with_confidence(text, model_name)
        
        if result[0] is not None:
            conf1_scores.append(result[0])
            conf2_scores.append(result[1])
            valid_indices.append(idx)
            explanation = result[2] if result[2] else "No explanation"
            print(f"Row {idx+1}/{total_rows}: Class 1: {result[0]}, Class 2: {result[1]} - {explanation}")
        else:
            print(f"Row {idx+1}/{total_rows}: Failed to get prediction")
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
    
    # Create output DataFrame with original data plus new columns
    output_df = df.copy()
    output_df['class_1_confidence'] = None
    output_df['class_2_confidence'] = None
    
    # Fill in the confidence scores for valid predictions
    for i, idx in enumerate(valid_indices):
        output_df.loc[idx, 'class_1_confidence'] = conf1_scores[i]
        output_df.loc[idx, 'class_2_confidence'] = conf2_scores[i]
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description='Test Martian GPT with confidence scores')
    parser.add_argument('--model', type=str, default='openai/gpt-4.1-nano')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    
    args = parser.parse_args()
    
    if not MARTIAN_API_KEY:
        print("Error: Please set MARTIAN_API_KEY in the script")
        return
    
    # Run the test
    result_df = test_martian_confidence_on_csv(args.model, args.input_csv, args.output_csv)
    
    print(f"Results saved to {args.output_csv}")
    print(f"Processed {len(result_df)} tweets")

if __name__ == '__main__':
    main()

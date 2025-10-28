#!/usr/bin/env python3
"""
Test Martian GPT models on the gold test set for tweet classification.
"""

import pandas as pd
import requests
import json
import time
from datetime import datetime
import os
import argparse

# Martian API configuration
MARTIAN_API_KEY = ""
MARTIAN_URL = "https://api.withmartian.com/v1/chat/completions"

def classify_tweet_with_martian(text, model_name, api_key=MARTIAN_API_KEY):
    """
    Classify a single tweet using specified Martian model
    """
    
    prompt = f"""Based on Kingdon's theory could you please classify this tweet into one of these categories:

1. Problem Oriented - The tweet describes or mentions a problem, issue, or challenge
2. Solution Oriented - The tweet describes or mentions a solution, fix, or resolution  
3. Political - The tweet is political in nature

Respond with ONLY the number (1, 2, or 3) corresponding to the category. No explanation needed. Also you seem to overclassify tweets as political, so please be more careful.

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
        "temperature": 0.1,  # Low temperature for consistent classification
        "max_tokens": 10
    }
    
    try:
        response = requests.post(MARTIAN_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        prediction_text = result['choices'][0]['message']['content'].strip()
        
        # Extract the number from the response
        try:
            # Look for numbers 1, 2, or 3 in the response
            for char in prediction_text:
                if char in ['1', '2', '3']:
                    return int(char)
            # If no number found, return 3 (Political) as default
            return 3
        except:
            return 3
            
    except Exception as e:
        return None

def test_martian_on_gold_test(model_name):
    """
    Test Martian GPT on the entire gold test set
    """
    # Load the test data
    test_df = pd.read_csv('tweets_data/GOLD_TEST.csv')
    
    # Initialize results
    predictions = []
    true_labels_filtered = []
    processing_times = []
    valid_indices = []
    
    # Process each tweet
    for idx, row in test_df.iterrows():
        text = row['text']
        true_label = row['AR']
        
        start_time = time.time()
        result = classify_tweet_with_martian(text, model_name)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        if result is not None:
            predictions.append(result)
            true_labels_filtered.append(true_label)
            valid_indices.append(idx)
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
    
    from sklearn.metrics import accuracy_score
    
    if len(predictions) > 0:
        accuracy = accuracy_score(true_labels_filtered, predictions)
    else:
        accuracy = 0.0
    
    from collections import Counter
    import numpy as np
    
    correct_predictions = (np.array(true_labels_filtered) == np.array(predictions))
    
    # Per-class accuracy
    class_accuracies = {}
    for class_id in [1, 2, 3]:
        class_mask = (np.array(true_labels_filtered) == class_id)
        if class_mask.sum() > 0:
            class_accuracy = correct_predictions[class_mask].mean()
            class_accuracies[class_id] = class_accuracy
        else:
            class_accuracies[class_id] = 0.0
    
    # Performance stats
    avg_time = sum(processing_times) / len(processing_times)
    total_time = sum(processing_times)
    
    results_df = pd.DataFrame({
        'tweet_id': [test_df.iloc[idx]['tweet_id'] for idx in valid_indices],
        'text': [test_df.iloc[idx]['text'] for idx in valid_indices],
        'true_label': true_labels_filtered,
        'predicted_label': predictions,
        'processing_time': [processing_times[idx] for idx in valid_indices]
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"martian_gpt_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    
    summary = {
        'timestamp': timestamp,
        'model': model_name,
        'total_samples': len(test_df),
        'valid_samples': len(predictions),
        'skipped_samples': len(test_df) - len(predictions),
        'accuracy': float(accuracy),
        'class_accuracies': class_accuracies,
        'avg_processing_time': avg_time,
        'total_processing_time': total_time
    }
    
    summary_file = f"martian_gpt_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("Per-Class Accuracy:")
    for class_id in [1, 2, 3]:
        count = (np.array(true_labels_filtered) == class_id).sum()
        acc = class_accuracies[class_id]
        print(f"  Class {class_id}: {acc:.4f} ({acc*100:.2f}%) - {count} samples")
    print("=" * 60)
    
    return results_df, summary

def main():
    parser = argparse.ArgumentParser(description='Test Martian GPT models on gold test set')
    parser.add_argument('--model', type=str, default='openai/gpt-4.1-nano'
    parser.add_argument('--test_csv', type=str, default='tweets_data/GOLD_TEST.csv'
    
    args = parser.parse_args()
    
    try:
        results_df, summary = test_martian_on_gold_test(args.model)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

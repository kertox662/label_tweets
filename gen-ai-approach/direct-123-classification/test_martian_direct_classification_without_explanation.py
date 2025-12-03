#!/usr/bin/env python3
import pandas as pd
import requests
import time
import argparse
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

MARTIAN_URL = "https://api.withmartian.com/v1/chat/completions"
MARTIAN_API_KEY = ""

def classify_tweet_direct_without_explanation(text, model_name, api_key=MARTIAN_API_KEY):
    """
    Directly classify a tweet as 1, 2, or 3 without explanation
    """
    
    prompt = f"""Based on Kingdon's theory, please classify this tweet into one of these categories:

1. Problem Oriented - The tweet describes or mentions a problem, issue, or challenge
2. Solution Oriented - The tweet describes or mentions a solution, fix, or resolution  
3. Political - The tweet is political in nature but doesn't clearly focus on problems or solutions

Respond with ONLY the number (1, 2, or 3) corresponding to the category.

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
        "max_tokens": 50
    }

    try:
        response = requests.post(MARTIAN_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        prediction_text = result['choices'][0]['message']['content'].strip()

        # Parse the response format: NUMBER
        try:
            # Extract the number (first token)
            pred = int(prediction_text.split()[0])
            
            # Validate the prediction
            if pred in [1, 2, 3]:
                return pred
        except Exception as parse_err:
            pass
        
        return None

    except Exception as e:
        print(f"API Error: {e}")
        return None

def test_direct_classification_without_explanation(model_name, input_csv, output_csv):
    """
    Test direct classification without explanation on the dataset
    """
    # Load the input data
    df = pd.read_csv(input_csv)
    
    # Initialize results
    predictions = []
    valid_indices = []
    total_rows = len(df)
    
    # Process each tweet
    for idx, row in df.iterrows():
        text = row['text']
        
        result = classify_tweet_direct_without_explanation(text, model_name)
        
        if result is not None:
            predictions.append(result)
            valid_indices.append(idx)
            print(f"Row {idx+1}/{total_rows}: Predicted: {result}")
        else:
            print(f"Row {idx+1}/{total_rows}: Failed to get prediction")
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
    
    # Create output DataFrame with original data plus new columns
    output_df = df.copy()
    output_df['predicted_class'] = None
    
    # Fill in the predictions for valid predictions
    for i, idx in enumerate(valid_indices):
        output_df.loc[idx, 'predicted_class'] = predictions[i]
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    
    # Calculate metrics if true labels are available
    if 'AR' in df.columns:
        valid_df = output_df.iloc[valid_indices]
        true_labels = valid_df['AR'].astype(float)
        pred_labels = valid_df['predicted_class'].astype(float)
        
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        
        print("\nPerformance Metrics:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        
        # Class-wise metrics
        report = classification_report(true_labels, pred_labels, output_dict=True)
        for cls in sorted(report.keys()):
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = report[cls]
                print(f"Class {cls}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
        
        # Save metrics to a separate CSV
        metrics_df = pd.DataFrame([{
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_class1': report['1.0']['f1-score'] if '1.0' in report else 0,
            'f1_class2': report['2.0']['f1-score'] if '2.0' in report else 0,
            'f1_class3': report['3.0']['f1-score'] if '3.0' in report else 0,
            'valid_predictions': len(valid_indices),
            'total_samples': len(df)
        }])
        metrics_df.to_csv(output_csv.replace('.csv', '_metrics.csv'), index=False)
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description='Test direct classification without explanation')
    parser.add_argument('--model', type=str, default='openai/gpt-4.1-nano')
    parser.add_argument('--input_csv', type=str, default='tweets_data/GOLD_TEST.csv')
    parser.add_argument('--output_csv', type=str, default='direct_classification_without_explanation_results.csv')
    
    args = parser.parse_args()
    
    if not MARTIAN_API_KEY:
        print("Error: Please set MARTIAN_API_KEY in the script")
        return
    
    # Run the test
    result_df = test_direct_classification_without_explanation(args.model, args.input_csv, args.output_csv)
    
    print(f"Results saved to {args.output_csv}")
    print(f"Processed {len(result_df)} tweets")

if __name__ == '__main__':
    main()

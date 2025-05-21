import pandas as pd
import pyreadr

df = pd.read_csv('final_combined_labelled.csv')
raw_df = pyreadr.read_r('tweets/raw_tweets.Rdata')['raw_tweets']
raw_df['tweet_id'] = raw_df['tweet_id'].astype(int)
raw_df['tweet_text'] = raw_df['text']
merged_df = df.merge(raw_df, on='tweet_id')
merged_df = merged_df[['tweet_id', 'tweet_text', 'AR', 'MB']]
sample = merged_df.sample(200)
sample.to_csv('sample_data.csv')
import pandas as pd
import os
if __name__ == "__main__":
    path = 'tweets/Labelled_Tweets/'

    success = pd.read_csv(path + '/success_new.csv')
    failed = pd.read_csv(path + '/failed_new.csv')

    success['tweet_id'] = success['tweet_id_y']
    success['old_tweet_id'] = success['tweet_id_x']
    success['author_id'] = success['author_id_y']
    success['old_author_id'] = success['author_id_x']
    success['tw_date'] = success['tw_date_y']
    success['text'] = success['text_y']
    success['old_text'] = success['text_x']

    failed['tweet_id'] = failed['tweet_id_y']
    failed['old_tweet_id'] = failed['tweet_id_x']
    failed['author_id'] = failed['author_id_y']
    failed['old_author_id'] = failed['author_id_x']
    failed['tw_date'] = failed['tw_date_y']
    failed['text'] = failed['text_y']
    failed['old_text'] = failed['text_x']

    success_final = success[['tweet_id', 'old_tweet_id', 'text', 'old_text', 'author_id', 'old_author_id', 'tw_date', 'year', 'AR', 'MB', 'similarity']]
    success_final.to_csv('success_master.csv', index=False)

    failed_final = failed[['tweet_id', 'old_tweet_id', 'text', 'old_text', 'author_id', 'old_author_id', 'tw_date', 'year', 'AR', 'MB', 'similarity']]
    failed_final.to_csv('failure_master.csv', index=False)
    final_df = success_final[['tweet_id', 'year', 'AR', 'MB']]
    final_df.to_csv('final_combined_labelled.csv', index=False)
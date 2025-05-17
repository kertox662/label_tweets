import pandas as pd
import pyreadr
from datetime import datetime


def __main__(self):
    raw_tweets = pyreadr.read_r('raw_tweets.RData')
    marked_tweets = pd.read_excel('2019_updated.xls')
    marked_tweets2 = pd.read_excel('Michelle.xlsx')

    marked_tweets = marked_tweets[['id', 'text', 'author_id']]
    marked_tweets['year'] = 2019
    marked_tweets['id'] = marked_tweets['id'].astype(float).astype(int)
    marked_tweets2['id'] = marked_tweets2['id'].astype(float).astype(int)
    marked_tweets2 = marked_tweets2[['id', 'text', 'author_id', 'year']]
    labelled_tweets = pd.concat([marked_tweets, marked_tweets2])

    # We dont want to sample years that you already labelled
    dont_use_years = list(str(x) for x in labelled_tweets['year'].unique())


    raw_df = raw_tweets['raw_tweets']
    # Only english tweets
    raw_df = raw_df[raw_df['lang']=='en']


    raw_df = raw_df[['tweet_id', 'text', 'author_id', 'tw_date']]

    # Grab the year as a string
    raw_df['tw_date'] = pd.to_datetime(raw_df['tw_date'], errors='coerce')
    raw_df['year'] = raw_df['tw_date'].dt.year.astype(str)

    # Remove invalid years
    raw_df = raw_df[raw_df['year'] != 'nan']

    # Convert to string nice format
    raw_df['year'] = raw_df['year'].astype(float).astype(int).astype(str)

    # Only look at 2012->2022
    years = [str(year) for year in range(2012, 2023)]
    # Check for labelled years and remove and only 2012->2023
    raw_df = raw_df[~raw_df['year'].isin(dont_use_years)]
    raw_df = raw_df[raw_df['year'].isin(years)]

    # Get rid of low contributing authors
    filtered_authors = raw_df['author_id'].value_counts()
    filtered_authors = filtered_authors[filtered_authors > 500].index

    # Remove those authors
    raw_df = raw_df[raw_df['author_id'].isin(filtered_authors)]
    raw_df = raw_df[raw_df['author_id'].isin(list(str(x) for x in raw_df['author_id'].unique()))]

    sampled_dataset = pd.DataFrame()
    author_count = raw_df['author_id'].value_counts()

    # Split into equal proportions of low, med, high authors
    bins = pd.qcut(author_count, q=3, labels=['low', 'med', 'high']).reset_index()
    bins['bin'] = bins['count']
    sample = raw_df.merge(bins, on='author_id', how='left')

    high_tweet = sample[sample['bin'] == 'high']
    med_tweet = sample[sample['bin'] == 'med']
    low_tweet = sample[sample['bin'] == 'low']

    # Sampling based on how many tweets we want from that author
    def sample_per_author(df, n):
        return df.groupby('author_id', group_keys=False).apply(lambda x: x.sample(n=min(len(x), n), random_state=42))

    # 15 per high 10 per medium and 5 per low
    high_tweet_sampled = sample_per_author(high_tweet, 15)
    med_tweet_sampled = sample_per_author(med_tweet, 10)
    low_tweet_sampled = sample_per_author(low_tweet, 5)

    final_sampled = pd.concat([high_tweet_sampled, med_tweet_sampled, low_tweet_sampled])

    # Counts
    print(final_sampled['year'].value_counts())
    print(final_sampled.shape)



    final_sampled = final_sampled[['tweet_id', 'text', 'author_id', 'tw_date', 'year']]

    # Save each file per year
    for year in final_sampled['year'].unique():
        # current = final_sampled[final_sampled['year'] == year]
        # current.to_csv(f'random_tweets_{year}.csv')
        pass





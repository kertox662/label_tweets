import pandas as pd
import os

def __main__(self):
    # path = os.path.expanduser('') # Change to location of downloaded files
    path = ""
    dfs = []

    for file in os.listdir(path):
        if 'Done' in file:
            full_path = os.path.join(path, file)
            df = pd.read_csv(full_path)
            df = df[['tweet_id', 'text', 'author_id', 'tw_date', 'year', 'AR', 'MB']]
            dfs.append(df)


    data = pd.concat(dfs, ignore_index=True)

    data.to_csv(path + 'combined_labelled.csv')
    data = pd.read_csv(path + 'combined_labelled.csv')
    other_data = pd.read_csv(path + '2019_updated.csv')
    other_data_2 = pd.read_csv(path + 'Michelle.csv')
    df_2019 = other_data[['tweet_id', 'text', 'author_id', 'tw_date', 'year', 'AR', 'MB']]
    df_2017_20 = other_data_2[['tweet_id', 'text', 'author_id', 'tw_date', 'year', 'AR', 'MB']]

    final_df = pd.concat([data, df_2019, df_2017_20], ignore_index=True)
    final_df = final_df[['tweet_id', 'year', 'AR', 'MB']]
    final_df['AR'] = final_df['AR'].astype('Int64')
    final_df.to_csv(path + 'final_combined_labelled.csv', index=False)
import pandas as pd


df = pd.read_csv("success_master.csv")

sampled = df.groupby('year', group_keys=False).apply(lambda x: x.sample(n=50, random_state=42))

train = df[~df.index.isin(sampled.index)]

# Save
sampled.to_csv("test_master.csv", index=False)
train.to_csv("train_master.csv", index=False)
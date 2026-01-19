import pandas as pd


df = pd.DataFrame({
    'A': [1, 2, 1, 4, 5],
    'B': ['a', 'b', 'a', 'd', 'e']
})

df.to_csv('sample_dataset.csv', index=False)
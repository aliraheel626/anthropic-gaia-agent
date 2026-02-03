import pandas as pd

df = pd.read_parquet('data/gaia/2023/validation/metadata.parquet')
mask = df['Question'].str.contains('Pie Menu', case=False, na=False)
result = df[mask]

for idx, row in result.iterrows():
    print('Question:')
    print(row['Question'])
    print('\nFinal Answer:', row['Final answer'])

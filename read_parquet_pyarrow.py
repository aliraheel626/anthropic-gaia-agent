import pyarrow.parquet as pq

# Read the parquet file
table = pq.read_table(r'C:\Users\User\Documents\GitHub\agent-sdk\data\gaia\2023\validation\metadata.parquet')

# Convert to pandas for easier filtering
df = table.to_pandas()

# Search for rows containing 'Pie Menu' in the Question column
mask = df['Question'].str.contains('Pie Menu', case=False, na=False)
result = df[mask]

if len(result) > 0:
    print('Found', len(result), 'row(s) containing "Pie Menu"')
    print('\n' + '='*80)
    for idx, row in result.iterrows():
        print(f'\nRow Index: {idx}')
        print(f'\nQuestion:\n{row["Question"]}')
        print(f'\nFinal Answer: {row["Final answer"]}')
        print('\n' + '='*80)
else:
    print('No rows found containing "Pie Menu"')

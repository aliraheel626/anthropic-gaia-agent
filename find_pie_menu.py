#!/usr/bin/env python3
import sys
sys.path.insert(0, r'C:\Users\User\Documents\GitHub\agent-sdk\.venv\Lib\site-packages')

import pandas as pd

try:
    # Read the parquet file
    parquet_file = r'C:\Users\User\Documents\GitHub\agent-sdk\data\gaia\2023\validation\metadata.parquet'
    df = pd.read_parquet(parquet_file)

    # Search for rows containing 'Pie Menu' in the Question column
    mask = df['Question'].str.contains('Pie Menu', case=False, na=False)
    result = df[mask]

    if len(result) > 0:
        print(f'Found {len(result)} row(s) containing "Pie Menu"\n')
        print('='*80)
        for idx, row in result.iterrows():
            print(f'\nRow Index: {idx}')
            print(f'\nQuestion:')
            print(row['Question'])
            print(f'\nFinal Answer: {row["Final answer"]}')
            print('\n' + '='*80)
    else:
        print('No rows found containing "Pie Menu"')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

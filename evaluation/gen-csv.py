#!/usr/bin/env python

from glob import glob
import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print(f'usage: python3 {sys.argv[0]} <base-dir>', file=sys.stderr)
        sys.exit(1)

    base_dir = sys.argv[1]
    entries = []

    for result in glob(f'{base_dir}/*_scored.csv'):
        data = pd.read_csv(result)
        model = data.iloc[0]['model_name']
        # assert model, f'model is None file \'{result}\''
        results = data.groupby('category')['score'].mean()
        entries.append(
            { 'model': model } | { key : round(((value - 1) / 4) * 100, 2) for key, value in results.items()}
        )

    frame = pd.DataFrame(data = entries)
    frame.set_index('model', inplace=True)
    frame.to_csv('alba-results.csv')

if __name__ == '__main__':
    main()

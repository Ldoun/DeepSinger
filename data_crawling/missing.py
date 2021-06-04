import os
import sys
import pandas as pd

music_dir = sys.argv[1]
tsv_file = sys.argv[2]

files = os.listdir(music_dir)
tsv = pd.read_csv(tsv_file,sep='\t')

cnt = 0
for file in tsv['titles']:
    if file not in files:
        print(1)
        cnt += 1

print(cnt)
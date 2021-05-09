import os
import sys
import pandas
import shutil

data_f = sys.argv[0]
src_f = sys.argv[1]
dst_f = sys.argv[2]

data = pd.read_csv(data_f,sep='\t')
for vid in data['video_name']:
    if not os.path.isfile(dst_f+vid+'.wav'):
        shutil.copy(src_f+vid+'.wav',dst_f)


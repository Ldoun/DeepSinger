import pandas as pd
from spleeter.separator import Separator

data = pd.read_csv('data/duration_filtered.live_rm.duet_rm.feat_rm.path_renamed.rm_missmatch.group_singer_rm.ipa.singer_up40.mr_removed.lyrics_alinged.tab_removed.tsv',sep='\t')
separator = Separator('spleeter:2stems')

# List of input to process.
audio_descriptors =['G:/내 드라이브/data/svs/wave_appropriate_data/'+name+'.wav' for name in data['video_name']]

# Batch separation export.
for index,x in enumerate(audio_descriptors):
    separator.separate_to_file(x, 'G:/내 드라이브/data/svs/vocal/'+str(index), synchronous=False)

# Wait for batch finish.a
separator.join()
# Use audio loader explicitly for loading audio waveform :
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from scipy.io.wavfile import write
import os
import sys
import time

music_dir = sys.argv[1]
# Using embedded configuration.
separator = Separator('spleeter:2stems')

audio_loader = AudioAdapter.default()
sample_rate = 44100
with open('diff.txt','r') as f:
    temp = f.readlines()
    files = [file.strip() for file in temp]

for index,v_name in enumerate(files):
    if os.path.isfile(os.path.join(music_dir,'vocal',v_name)):
        continue

    start = time.time()
    try:
        waveform, _ = audio_loader.load(os.path.join(music_dir,v_name), sample_rate=sample_rate)

        # Perform the separation :
        prediction = separator.separate(waveform)
    except Exception as e:
        print(e)
        print(index)
        continue

    write(os.path.join(music_dir,'vocal',v_name),sample_rate,prediction['vocals'])
    write(os.path.join(music_dir,'accompaniment',v_name),sample_rate,prediction['accompaniment'])
    end = time.time()

    print(str(index)+'/'+str(len(files))+'\t'+start-end)


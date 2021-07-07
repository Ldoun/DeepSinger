import librosa
import pandas as pd
import os
import sys
import numpy as np
from scipy.io.wavfile import write

music_dir = sys.argv[1]
output_dir = sys.argv[2]

files = os.listdir(music_dir)
for index,m_file in enumerate(files):
    print(len(files) - index)  
    if os.path.isfile(os.path.join(output_dir,m_file)):
        continue

    a,sr = librosa.load(os.path.join(music_dir,m_file))

    non_vocal_cnt = 0
    droped_index = []
    for i,frame in enumerate(a):
        if frame > 0.1:
            if non_vocal_cnt > 10:
                removed_a = np.append([0,0,0,0,0,0,0,0,0,0],np.delete(a,droped_index))
                break
        else:
            droped_index.append(i)
            non_vocal_cnt += 1

    write(os.path.join(output_dir,m_file),sr,removed_a)

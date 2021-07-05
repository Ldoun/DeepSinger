import librosa
import sys
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import numpy as np
import parselmouth
import math

music_dir = sys.argv[1]
output_dir = sys.argv[2]

files = os.listdir(music_dir)
for index,m_file in enumerate(files):
    print(len(files) - index)
    if os.path.isfile(os.path.join(output_dir,m_file)):
        continue

    voice, sr= librosa.load(os.path.join(music_dir,m_file),sr=22050)
    snd = parselmouth.Sound(voice)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']

    droped_index = []
    for i,p in enumerate(pitch_values):
        index = i + 1
        if index == 1 and p ==0:
            droped_index = list(range(0,220))
        if p == 0:
            if index % 2 == 0:
                droped_index += list(range(math.floor(220.5 * index) , math.floor(220.5 * index) + 220 ))
            if index % 2 == 1:
                droped_index += list(range(math.floor(220.5 * index),math.floor(220.5 * index) + 221 ))

    start = droped_index[0] 
    pre = droped_index[0]
    result = []
    for i in droped_index[1:]:
        if pre + 1  == i:
            pre = i
            continue
        else:
            voice[start:pre] = [0 * 10]
            result.append([start,pre])
            start = i
            pre = i

    voice[start:pre] = [0 * 10]
    write(os.path.join(output_dir,m_file),22050,voice)
import sys
from scipy.io.wavfile import write
import os
import librosa
import math
import numpy as np

import pyloudnorm as pyln
import parselmouth
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

music_dir = sys.argv[1]
output_dir = sys.argv[2]
thresh_hold = sys.argv[2]

audio_loader = AudioAdapter.default()
separator = Separator('spleeter:2stems')

thresh_hold = float(thresh_hold)
files = os.listdir(music_dir)
for index,m_file in enumerate(files):
    print(len(files) - index)
    if os.path.isfile(os.path.join(output_dir,m_file)):
        continue

    #vocal extraction
    try:
        waveform, _ = audio_loader.load(os.path.join(music_dir,m_file), sample_rate=22050)
        prediction = separator.separate(waveform)
    except Exception as e:
        print(e)
        
    y = np.swapaxes(prediction['vocals'], 0, 1)
    y = librosa.to_mono(y)

    meter = pyln.Meter(22050) 
    loudness = meter.integrated_loudness(y)

    #normalize loudness
    loudness_normalized_audio = pyln.normalize.loudness(y, loudness, -16.0)

    snd = parselmouth.Sound(loudness_normalized_audio)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    

    fitered_voice = np.array([])
    non_vocal_frame_cnt = 0
    frames_sum = []
    if pitch_values[0] ==0 or sum(abs(loudness_normalized_audio[:441])) <= 441*thresh_hold:
        non_vocal_frame_cnt += 1
    else:
        fitered_voice = np.append(fitered_voice ,loudness_normalized_audio[:441])

    for i,p in enumerate(pitch_values):
        index = i + 1 
        frame_cnt = 441 + (index % 2)
        frame_sum = sum(abs(loudness_normalized_audio[math.floor(441 * index):math.floor(441 * index) + frame_cnt]))
        frames_sum.append(frame_sum)
        if p == 0 or frame_sum < frame_cnt*thresh_hold:
            non_vocal_frame_cnt += 1
        
        else:
            if non_vocal_frame_cnt > 10:
                fitered_voice = np.append(fitered_voice,np.zeros(4410))
                
            fitered_voice = np.append(fitered_voice ,loudness_normalized_audio[math.floor(441 * index):math.floor(441 * index) + frame_cnt])
            non_vocal_frame_cnt = 0
    
    write(os.path.join(output_dir,m_file),22050,fitered_voice)

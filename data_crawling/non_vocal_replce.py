import librosa
import sys
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

music_dir = sys.argv[1]
output_dir = sys.argv[2]

files = os.listdir(music_dir)
for m_file in files:
    if os.path.isfile(os.path.join(output_dir,m_file)):
        continue

    voice, sr= librosa.load(os.path.join(music_dir,m_file),sr=22050)

    fitered_voice = []
    sl_frame_cnt = 0
    rp_frame = []
    for index,frame in enumerate(voice):
        if abs(frame) < 0.0001:
            sl_frame_cnt += 1
        else:
            if sl_frame_cnt > 10:
                rp_frame.append(index)
                for i in range(10):
                    fitered_voice.append(0)
                
            fitered_voice.append(frame)
            sl_frame_cnt = 0

    fitered_voice = np.array(fitered_voice)
    write(os.path.join(output_dir,m_file),sr,fitered_voice)
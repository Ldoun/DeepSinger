#-16 LUFS
import soundfile as sf
import pyloudnorm as pyln
import sys
from scipy.io.wavfile import write
import os

music_dir = sys.argv[1]
output_dir = sys.argv[2]

files = os.listdir(music_dir)
for index,m_file in enumerate(files):
    print(len(files) - index)
    if os.path.isfile(os.path.join(output_dir,m_file)):
        continue

    data, rate = sf.read(os.path.join(music_dir,m_file)) # load audio

    # measure the loudness first  y
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -16.0)

    write(os.path.join(output_dir,m_file),rate,loudness_normalized_audio)

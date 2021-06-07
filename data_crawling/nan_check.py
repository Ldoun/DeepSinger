import os
import sys
import torchaudio
import torch

music_dir = sys.argv[1]

for file in os.listdir(music_dir):
    tensor,_ = torchaudio.load(os.path.join(music_dir,file))
    if torch.isnan(tensor).any():
        print(file)



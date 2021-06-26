import argparse
import sys
import pandas as pd
import os

import torch
import numpy as np
import torchaudio
from  torch.cuda.amp import autocast
from alignment.utils import detach_hidden

from alignment.dataloader import LJSpeechDataset,RandomBucketBatchSampler,TextAudioCollate
from alignment.Tokenizer import tokenizer
from alignment.model.lyrics_alignment import alignment_model

from scipy.io.wavfile import write
from torchaudio import save,transforms
from librosa.feature.inverse import mel_to_audio


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )

    p.add_argument(
        '--bpe_model',
        required=True,
        help='bpe_model file name'
    )

    p.add_argument(
        '--music_dir',
        required=True,
        help='music directory'
    )

    p.add_argument(
        '--new_music_dir',
        required=True,
        help='music directory where new music file will be save'
    )

    p.add_argument(
        '--data_fn',
        required=True,
        help='tsv data for inference'
    )

    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--multi_gpu',
        action='store_true',
        help='multi-gpu',
    )

    config = p.parse_args()
    return config

def get_model(input_size, output_size, train_config):
    model = alignment_model(
            input_size,
            output_size,
            train_config.word_vec_size,  
            train_config.en_hs,
            train_config.de_hs,
            train_config.attention_dim,
            train_config.location_feature_dim,
            train_config.dropout 
        )
   

    return model

def load_audio(file_f):
    audio_transformer = torchaudio.transforms.MelSpectrogram(sample_rate=22050,n_fft=1024,hop_length=256,normalized=True)
    audio,sr = torchaudio.load(f'{file_f}.wav')
    
    audio = audio_transformer(audio)
    
    return audio.squeeze(0)

def split_lyrics(lyric,seperation_mark):
    lyrics = np.array(lyric)
    seperation_frame = np.where(lyric == seperation_mark)
    splited_lyric = []
    for i in range(len(seperation_frame) -1):
        split_lyrics.append(lyrics[seperation_frame[i]:seperation_frame[i+1]])

    return split_lyrics

if __name__ == '__main__':
    config = define_argparser()
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu',
    )

    train_config = saved_data['config']

    tok = tokenizer(config.bpe_model)

    data = pd.read_csv(f'{config.data_fn}', sep='\t',
                                    usecols=['video_name', 'lyrics'],
                                    )

    input_size, output_size = 128, len(tok.vocab)

    model = get_model(input_size,output_size,train_config)
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    sr = 22050

    new_video_name = []
    new_lyrics = []

    inverse_mel = transforms.InverseMelScale(sample_rate=22050,n_stft=1024//2+1)
    griffin_lim = transforms.GriffinLim(n_fft=1024,hop_length=256)

    with torch.no_grad():
        device = next(model.parameters()).device
        for input_data in data.iterrows():
            x = load_audio(os.path.join(config.music_dir,input_data[1]['video_name'])).unsqueeze(0)
            y = np.array(tok.get_idx(input_data[1]['lyrics']))

            input_y = torch.IntTensor(y)[:-1].unsqueeze(0)
            x_length = x.size(2)
            y_length = len(y)

            lyrics = split_lyrics(y[1:-1],tok.seperation_mark[0])
            print('seperation_mark',tok.seperation_mark)
            print('y_length',y_length)

            cnt = 0
            last_index = 0
            chunk_index = 0
            start_index = np.zeros((x.size(0),), dtype=int)
            attention_index = 0
            encoder_hidden = None
            
            while chunk_index < y_length -1:  
                model.eval()

                with autocast():
                    chunk_y = input_y[:,chunk_index:chunk_index + train_config.tbtt_step].to(device)
                    chunk_y_label = y[chunk_index:chunk_index + train_config.tbtt_step]
                    
                    start_index = start_index + attention_index
                    
                    if encoder_hidden is None:
                        chunk_x = x[:,:,start_index[0] : start_index[0] + train_config.tbtt_step * (x_length // y_length)].to(device)

                        y_hat,mini_attention,encoder_hidden,decoder_hidden = model((chunk_x,None),chunk_y)
                    else:
                        chunk_x = x[:,:,start_index[0] : start_index[0] + train_config.tbtt_step * (x_length // y_length)].to(device)
                        encoder_hidden = detach_hidden(encoder_hidden)
                        decoder_hidden = detach_hidden(decoder_hidden)
                        y_hat,mini_attention,encoder_hidden,decoder_hidden = model((chunk_x,None),chunk_y,en_hidden = encoder_hidden,de_hidden = decoder_hidden)# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40
                    
                    attention_index = np.array(torch.argmax(mini_attention[:,-1,:],dim=1).tolist())
                    chunk_index = chunk_index + train_config.tbtt_step

                    if tok.seperation_mark in chunk_y_label:
                        for seperation_index in np.where(np.array(chunk_y_label) == '%')[0]:
                            seperation_frame = torch.argmax(mini_attention[:,seperation_index,:],dim = 1).item()

                            output = inverse_mel(x[:,:,last_index:seperation_frame])
                            output = griffin_lim(output)
                            save(os.path.join(config.music_dir,input_data[1]['video_name'] + '_'+ str(cnt) +'.wav'),output,sr)

                            #write(os.path.join(config.music_dir,input_data[1]['video_name'] + '_'+ str(cnt) +'.wav'),sr,mel_to_audio(np.array(x[:,:,last_index:seperation_frame].unsqueeze(0).tolist()),sr=22050,n_fft=1024,hop_length=256))
                            last_index = seperation_frame
                            new_video_name.append(input_data[1]['video_name'] + '_'+ str(cnt) +'.wav')
                            new_lyrics.append(lyrics[cnt])
                            cnt += 1 
                            
            if chunk_y_label[-1] != '%':
                output = inverse_mel(x[:,:,last_index:])
                output = griffin_lim(output)
                save(os.path.join(config.music_dir,input_data[1]['video_name'] + '_'+ str(cnt) +'.wav'),output,sr)

                #write(os.path.join(config.music_dir,input_data[1]['video_name'] + '_'+ str(cnt) +'.wav'),sr,mel_to_audio(np.array(x[:,:,last_index:].unsqueeze(0).tolist()),sr=22050,n_fft=1024,hop_length=256))
                new_video_name.append(input_data[1]['video_name'] + '_'+ str(cnt) +'.wav')
                new_lyrics.append(lyrics[cnt])
                cnt += 1 
                            


    result = pd.DataFrame()
    result['lyric'] = pd.Series(new_lyrics)
    result['video_name'] = pd.Series(new_video_name)
    result.to_csv('result.csv',index=False,sep='\t')

            
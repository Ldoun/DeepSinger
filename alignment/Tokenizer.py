import pickle
import os
import pandas as pd
import sentencepiece as spm

class tokenizer(object):
    def __init__(self,bpe_model):
        super().__init__()
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.vocab = spm.SentencePieceProcessor(model_file=bpe_model)

        self.seperation_mark = self.vocab.Encode('%')  # if seperation mark is not '%' change it
        self.pho = 'rʲoɯʊɕsɐemtkʑɡɹqʒɪðhdfɔɛː%zjiabpuŋvɑɫɾʃ̩wɜθʌəlɒn'
    def get_idx(self,line):
        ids = self.vocab.encode_as_ids(line)
        ids = [self.vocab.bos_id()] + ids + [self.vocab.eos_id()]
        return ids

    def get_word(self,idx):
        result = self.vocab.decode(idx)
        return result


class pho_tokenizer(object):
    def __init__(self):
        super().__init__()
        pho = '%rʲoɯʊɕsɐemtkʑɡɹqʒɪðhdfɔɛːzjiabpuŋvɑɫɾʃ̩wɜθʌəlɒn '
        
        self.vocab = {s: i+3 for i, s in enumerate(pho)}
        self._id_to_symbol = {i+3: s for i, s in enumerate(pho)}

        self.pad = 0
        self.bos = 1
        self.eos = 2

        self.vocab['<pad>'] = self.pad
        self.vocab['<bos>'] = self.bos
        self.vocab['<eos>'] = self.eos

        self._id_to_symbol[self.pad] = '<pad>'
        self._id_to_symbol[self.bos] = '<bos>'
        self._id_to_symbol[self.eos] = '<eos>'

        self.unk = len(self.vocab)

    def get_idx(self,line):
        idx = [self.vocab.get(char, self.unk) for char in line]
        return [self.bos] + idx + [self.eos]

    def get_word(self,idx):
        return "".join(self._id_to_symbol.get(i, '<unk>') for i in idx)


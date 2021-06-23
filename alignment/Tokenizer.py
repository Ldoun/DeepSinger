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
    def get_idx(self,line):
        ids = self.vocab.encode_as_ids(line)
        ids = [self.vocab.bos_id()] + ids + [self.vocab.eos_id()]
        return ids

    def get_word(self,idx):
        result = self.vocab.decode(idx)
        return result


    
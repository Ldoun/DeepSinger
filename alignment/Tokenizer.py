import pickle
import os
import pandas as pd

class tokenizer(object):
    def __init__(self,v):
        super().__init__()
        self.eos = 2
        self.sos = 1
        self.pad = 0
        
        self.vocab = {'<PAD>':self.pad,'<SOS>':self.sos,'<EOS>':self.eos}
        if v is not None:
            self.vocab = vocab

        '''self.vocab_f = vocab_f
        if os.path.isfile(self.vocab_f):
            with open(self.vocab_f, 'rb') as fr:
                self.vocab = pickle.load(fr)'''
        
    def get_idx(self,words):
        if isinstance(words,list):
            result = ['<SOS>']
            for w in words:
                result.append(self.vocab[w])
            result.append('<EOS>')
        
        if isinstance(words,str):
            result = [self.sos]
            words = words.split(' ')
            for w in words:
                result.append(self.vocab[w])
            result.append(self.eos)
        else:
            print('wrong input type')
            raise
        
        return result

    def get_word(self,idx):
        if isinstance(idx,list):
            result = []
            for i in idx:
                list(self.vocab.keys())[list(self.vocab.values()).index(i)]
        else:
            list(self.vocab.keys())[list(self.vocab.values()).index(idx)]
        
        return result

    def make_vocab(self,sentences):
        print(sentences)
        for word in sentences.split(' '):
            if isinstance(word,str):
                if word not in self.vocab.keys():
                    self.vocab[word] = len(self.vocab)

    def save(self):
        with open(self.vocab_f,'wb') as f:
            pickle.dump(self.vocab, f)

    def set_vocab(self,tsv_file):
        metadata = pd.read_csv(f'{tsv_file}', sep='\t',
                                    usecols=['titles', 'lyrics'],
                                    ) 

        for i in metadata['lyrics'].values:
            self.make_vocab(i)
        
        print(self.vocab)

    
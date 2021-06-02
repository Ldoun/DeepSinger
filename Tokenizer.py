class tokenizer(object):
    def __init__(self):
        super().__init__()
        self.eos = 2
        self.sos = 1
        self.pad = 0
        
        self.vocab = {'<PAD>':self.pad,'<SOS>':self.sos,'<EOS>':self.eos}
        
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
               

    
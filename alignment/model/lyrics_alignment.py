import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import pickle

#1차 senetence level 분할 모델 학습
#2차 sentence level 분할 모델을 바탕으로 전이학습하듯 phoneme level 분할 학습을 진행함
# output 모델 구조는 동일하나 모델 파일 2개
#activation function 체크
#mask 적용
#mel start frame implement

#amp 오늘 안에 해결 안되면 Native torch로 변환


class  location_sensitive_attention(nn.Module):
    def __init__(self,encoder_hidden_size,decoder_hidden_size,attention_dim,location_feature_dim,attention_kernel_size):
        super().__init__()
        self.F = nn.Conv1d(in_channels=1, out_channels=location_feature_dim,
                            kernel_size=attention_kernel_size, stride=1, padding=int((attention_kernel_size - 1) / 2),
                            bias=False)
        
        self.W = nn.Linear(decoder_hidden_size, attention_dim, bias=False) # keep one bias
        self.V = nn.Linear(encoder_hidden_size, attention_dim, bias=False)
        self.U = nn.Linear(location_feature_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.reset()
    
    def reset(self):
        """Remember to reset at decoder step 0"""
        self.Vh = None # pre-compute V*h_j due to it is independent from the decoding step i
    
    def _cal_energy(self, query, values, cumulative_attention_weights, mask=None):
        """Calculate energy:
           e_ij = score(s_i, ca_i-1, h_j) = v tanh(W s_i + V h_j + U f_ij + b)
           where f_i = F * ca_i-1,
                 ca_i-1 = sum_{j=1}^{T-1} a_i-1
        Args:
            query: [N, 1, Hd], decoder state
            values: [N, Ti, He], encoder hidden representation
            cumulative_attention_weights: (batch_size, 1, max_time)
        Returns:
            energies: [N, Ti]
        """
        # print('query', query.size())
        # print('values', values.size())
        #query = query.unsqueeze(1) #[N, 1, Hd], insert time-axis for broadcasting
        #print(query.shape)
        Ws = self.W(query) #[N, 1, A]
        if self.Vh is None:
            self.Vh = self.V(values) #[N, Ti, A]
        #print(cumulative_attention_weights.shape)

        location_feature = self.F(cumulative_attention_weights) #[N, 32, Ti]
        # print(location_feature.size())
        Uf = self.U(location_feature.transpose(1, 2)) #[N, Ti, A]
        '''print('W s_i', Ws.size())
        print('V h_j', self.Vh.size())
        print('U f_ij', Uf.size())'''
        energies = self.v(torch.tanh(Ws + self.Vh + Uf)).squeeze(-1) #[N, Ti]
        
        # print('mask', mask)
        # print('energies', energies)
        if mask is not None:
            #print('energies',energies.shape)
            #print('mask',mask.shape)
            energies = energies.masked_fill(mask, -np.inf)

        # print(energies)
        return energies

    def forward(self, query, values, cumulative_attention_weights, mask=None):
        """
        Args:
            query: [N, Hd], decoder state
            values: [N, Ti, He], encoder hidden representation
            mask: [N, Ti]
        Returns:
            attention_context: [N, He]
            attention_weights: [N, Ti]
        """
        #print("values",values.shape)
        energies = self._cal_energy(query, values, cumulative_attention_weights, mask) #[N, Ti]
        attention_weights = F.softmax(energies, dim=1) #[N, Ti]
        # print('weights', attention_weights)
        #print('energies',energies.shape)
        #print("values",values.shape)
        #print("attention_weights",attention_weights)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), values) #[N, 1, Ti] bmm [N, Ti, He] -> [N, 1, He]
        #attention_context = attention_context.squeeze(1) # [N, Ti]
        # print('context', attention_context.size())
        return attention_context, attention_weights
    
class ConvolutionBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,padding):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=kernel_size,padding=padding)
        self.b_norm = nn.BatchNorm1d(num_features=out_ch)
        self.relu = nn.ReLU()


    #https://github.com/pytorch/pytorch/issues/1206#issuecomment-292440241

    def forward(self,x):
        a = x
        #print(torch.any(torch.any(x != 0,dim=2),dim=1))
        if torch.isnan(x).any():
            print('nan input')
            print(torch.isnan(x).any())

        for row in x:
            if torch.sum(row,dim=(0,1)) == 0:
                print('yes...')
        #print('input_x',torch.isnan(x).any())
        x = x.to(torch.float32)
        if torch.isnan(x).any():
            print('float32')
            print(torch.isnan(x).any())
        #print('float_x',torch.isnan(x).any())
        x = self.conv1d(x)
        #print('conv1d',torch.isnan(x).any())
        if torch.isnan(x).any():
            print('conv1d')
        x = self.b_norm(x)
        if torch.isnan(x).any():
            with open('makes_nan.pickle','wb') as f:
                pickle.dump(a,f)
                raise Exception('nan 발생')

            print('b_norm')
            print('*'*50)
            print(torch.any(torch.any(x != 0,dim=2),dim=1))
        #print(b_norm,x.shape)
        #print('b_norm',torch.isnan(x).any())
        x = self.relu(x)
        if torch.isnan(x).any():
            print('relu')
            print(torch.isnan(x).any())
        #print('relu',torch.isnan(x).any())
        return x

class mel_encoder(nn.Module):
    def __init__(self, in_ch = 128,out_ch=512,kernel=9,pad=4,drop_p=0.1):
        super().__init__()
        
        self.prenet = nn.Sequential(
            #3-layer conv1d hidden_size:512 kernel_size:9
            ConvolutionBlock(in_ch=in_ch,out_ch=out_ch,kernel_size=kernel,padding=pad),
            ConvolutionBlock(in_ch=out_ch,out_ch=out_ch,kernel_size=kernel,padding=pad),
            ConvolutionBlock(in_ch=out_ch,out_ch=out_ch,kernel_size=kernel,padding=pad),
        )

        #bi-diretional LSTM hidden_size:512
        self.rnn = nn.LSTM(
            input_size=out_ch,
            hidden_size=int(out_ch/2),
            batch_first=True,
            bidirectional=True,
            dropout = drop_p
        )

    def forward(self,mel,hidden,length = None):
        x = mel         
        #print('input_x',torch.isnan(x).any())
        x = self.prenet(x)  # (bs,128,length)  
        #print('prenet',x.shape) 

        #print('prenet',torch.isnan(x).any())
        x = x.transpose(1,2) #(bs,512,length)


        if length is not None: 
            x = pack(x,length,batch_first=True) # enforce_sorted = True tensor내 정렬 필요
            
        if hidden is None:
            #print('hidden none')
            y,h = self.rnn(x) #(bs,length,512)
        else:
            #print('hidden not none')
            y,h = self.rnn(x,hidden)

        #print('rnn',torch.isnan(y).any())
        #print('length',length)

        #print('rnn',y.shape)
        if length is not None:
            y, _ =  unpack(y,batch_first=True)
          
        return y,h #(bs,length,512)


class phoneme_decoder(nn.Module):
    def __init__(self,embedding_dim=512,encoder_hs = 512,hidden_size=1024,drop_p = 0.1):
        super().__init__()
        #2-layer LSTM hidden_size: 1024
        self.rnn = nn.LSTM(
            input_size=embedding_dim + encoder_hs,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout = drop_p
        )
    
    def forward(self,emb_tgt,attention_context_vector,h_t_1):
        #emb_tgt 이전 time step의 y의 임베딩 벡터
        #attention_context_vector encoder attention applied context vector
        #h_t_1 이전 타임스텝의 디코더 hs,cs
        
        batch_size = emb_tgt.size(0)
        hidden_size = h_t_1[0].size(-1)

        #if attention_context_vector is None:
            #first time-step
            #attention_context_vector = emb_tgt.new(batch_size,1,hidden_size).zero_()
            #해당 tensor와 같은 type,디바이스로 tensor를 만들어주는 함수

        #emb_tgt needs to be size(bs,512)
        x = torch.cat([emb_tgt,attention_context_vector],dim = -1)
        #|x| = (batch_size,1,word_vec_size + hidden_size)

        y,h = self.rnn(x,h_t_1)
        #|y| = (batch_size,1,hidden_size)
        #|h[0]| = (num_layers,batch_size,hidden_size)

        return y,h

class Generator(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()

        self.output = nn.Linear(input_size,out_features= output_size) 
        self.softmax = nn.LogSoftmax(dim = -1) 
        
    def forward(self,x):
        y = self.softmax(self.output(x))
        #|y| = (batch_size,length,ouput_size(vocab_size))
        return y

class alignment_model(nn.Module):
    def __init__(
        self,
        input_size=128,
        vocab_size = None,
        emb_hs=512,
        en_hidden_size=512,
        de_hidden_size=1024,
        attention_dim=256,
        location_feature_dim = 128,
        drop_p=0.1
    ):

        super().__init__()
        self.decoder_hs = de_hidden_size
        self.encoder_hs = en_hidden_size
        self.encoder = mel_encoder(in_ch = input_size,out_ch=en_hidden_size,kernel=9,pad=4,drop_p=drop_p)
        self.decoder = phoneme_decoder(embedding_dim=emb_hs,hidden_size=de_hidden_size,encoder_hs=en_hidden_size,drop_p = drop_p)
        self.attention = location_sensitive_attention(
                            encoder_hidden_size=en_hidden_size,
                            decoder_hidden_size=de_hidden_size,
                            attention_dim = attention_dim,
                            location_feature_dim = location_feature_dim,
                            attention_kernel_size = 31,
        )
        self.p_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_hs)
        self.concat = nn.Linear(en_hidden_size + de_hidden_size,1024)
        self.generator = Generator(input_size=1024,output_size=vocab_size)

    def generate_mask(self,x,length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [torch.cat([x.new_ones(1,l).zero_(),
                                    x.new_ones(1,(max_length - l))]
                                    ,dim = -1)]

            else:
                mask += [x.new_ones(1,l).zero_()]

        mask = torch.cat(mask,dim=0).bool()
        return mask
    

    def forward(self, mel,ipa, en_hidden = None,de_hidden = None):
        
        mask = None
        
        if isinstance(mel,tuple):
            x,mask = mel #torch text에서 x_length
            #|mask| = (batch_size,length)
        else:
            x = mel

        #|mel| = (bs,128,length)    
        #|ipa| = (bs,length,vocab)

        mel_length = x.size(2)

        if isinstance(ipa,tuple):
            ipa = ipa[0]
            text_mask = ipa[1]
        

        batch_size = ipa.size(0)

    
        h_src, encoder_hidden = self.encoder(x,en_hidden)
        #|h_src| = (batch_size,length,hidden_size)
        #|h_0_tgt| = (num_layers * 2,batch_size, hidden_size / 2)

        #h_0_tgt = self.fast_merge_encoder_hidden(h_0_tgt)
        emb_tgt = self.p_embedding(ipa)
        #print(emb_tgt.shape)
        #|emb_tgt| = (batch_size,length,word_vec_size)

        h_tilde = []
        attention = []
        
        '''if de_hidden is None:
            decoder_hidden = (
                h_src.new_zeros(2,h_src.size(0),self.decoder_hs),
                h_src.new_zeros(2,h_src.size(0),self.decoder_hs)
            )

        else:
            decoder_hidden = de_hidden'''

        decoder_hidden = encoder_hidden[0][0,:,:],encoder_hidden[0][0,:,:]
        decoder_output = decoder_hidden[0]
        print(decoder_output.shape)
        #attention_context_vector = emb_tgt.new_zeros(batch_size,1,self.encoder_hs)
        cumulative_attention_weights = h_src.new_zeros(h_src.size(0),mel_length)

        for t in range(ipa.size(1)):
            if t == 0:
               self.attention.reset()
                
            emb_t = emb_tgt[:,t,:].unsqueeze(1)
            #print(t)
            #print("emn_t:",emb_t.shape)
            #print("h_src:",h_src.shape)
            #emb_t = emb_t.unsqueeze(1)
            cumulative_attention = cumulative_attention_weights.unsqueeze(1)
            #|emb_t| = (batch_size,1,word_vec_size)
            #|h_t_tilde| = (batch_size,1,hidden_size)
            #|attention_context_vector| = (batch_size,1,encoder_hidden_size)
            #attention 구하고 decoder에 넣기
            attention_context_vector, attention_weights = self.attention(decoder_output,h_src,cumulative_attention,mask)
            cumulative_attention_weights = cumulative_attention_weights + attention_weights
            
            decoder_output,decoder_hidden = self.decoder(emb_t, attention_context_vector,decoder_hidden)

            #|decoder_output| = (batch_size,1,hidden_size)
            #|decoder_hidden| = (n_layer,batch_size,hidden_size)

            #print('decoder_output',decoder_output.shape)
            #print('attention_context_vector',self.attention_context_vector.shape)
            h_t_tilde = self.concat(torch.cat([decoder_output,attention_context_vector],dim=-1))
            #|h_t_tilde| = (batch_size,1,hidden_size)
            #print('h_t_tilde', h_t_tilde.shape)
            
            h_tilde += [h_t_tilde]
            
            attention += [attention_weights]
        
        #print('ipa',ipa.size())
        #print(h_tilde)
        h_tilde = torch.cat(h_tilde,dim=1)
        attention = torch.stack(attention,dim=1)
        #|h_t_tilde| = (batch_size,length,hidden_size)

        y_hat = self.generator(h_tilde)
        
        #|y_hat| = (batch_size,length,ouput_size)

        '''print('emb_tgt',torch.isnan(emb_tgt).any())
        print('cumulative_attention',torch.isnan(cumulative_attention).any())
        print('decoder_output',torch.isnan(decoder_output).any())
        print('h_src', torch.isnan(h_src).any())
        print('encoder_hidden', torch.isnan(encoder_hidden[0]).any())
        print('encoder_hidden', torch.isnan(encoder_hidden[1]).any())
        if en_hidden is not None:
            print('en_hidden0',torch.isnan(en_hidden[0]).any())
            print('en_hidden1',torch.isnan(en_hidden[1]).any())'''

        return y_hat,attention,encoder_hidden,decoder_hidden

        
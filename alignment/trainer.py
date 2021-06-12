from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from  torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from alignment.utils import get_grad_norm,get_parameter_norm,detach_hidden,guided_attentions,apply_attention_make_batch
from ignite.contrib.handlers.tensorboard_logger import *

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
import ignite.distributed as idist

#TODO
#TBTT 
#guided_attention loss
#attention map 합치기
#TBTT일때 pack_padded_sequence, masking 
#length 제거 -> pack padded sequence 제거


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MaximumLikelihoodEstimationEngine(Engine):
    def __init__(self, func, model, crit, optimizer,lr_scheduler,config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.max_target_ratio = 0.1
        super().__init__(func)

        self.best_loss =np.inf
        self.best_model = None
        self.scaler = GradScaler()
        #self.tbtt_step = tbtt_step

    @staticmethod
    def train(engine, mini_batch):
        if engine.config.multi_gpu:
            device = idist.device()
        else:
            device = next(engine.model.parameters()).device
        
        x,mask,x_length = mini_batch[0][0],mini_batch[0][1],mini_batch[0][2] #tensor,mask,length
        mini_batch_tgt = (mini_batch[1][0],mini_batch[1][1])
        y_length = mini_batch_tgt[1]
        
        y = mini_batch_tgt[0][:,1:]  #<BOS> 제외 정답문장 1번 단어부터 비교
        #|x| = (batch_size,128,length)
        #|y| = (batch_size,length)
        #print('x:',x.size())
        #print('y:',y.size())

        #print(x_length)
        #print('-' * 70)
        #print(y_length)

        encoder_hidden,decoder_hidden = None,None 
        loss_list = []
        chunk_index = 0
        start_index = np.zeros((x.size(0),), dtype=int)
        attention_index = 0
        input_y = mini_batch_tgt[0][:,:-1]
        #print(engine.max_target_ratio)
        
        with torch.autograd.set_detect_anomaly(True):
            with autocast():
                while chunk_index < engine.max_target_ratio * np.mean(y_length.tolist()):      
                    engine.model.train()
                    engine.optimizer.zero_grad()

                    chunk_y = input_y[:,chunk_index:chunk_index + engine.config.tbtt_step].to(device,non_blocking=engine.config.multi_gpu)
                    chunk_y_label = y[:,chunk_index:chunk_index + engine.config.tbtt_step].to(device,non_blocking=engine.config.multi_gpu)
                    
                    chunk_length = []
                    start_index = start_index + attention_index
                    
                    #print('start_index',start_index)
                    #print('attention_index',attention_index)
                        
                    #print('chunk_x:',chunk_x.shape)
                    if encoder_hidden is None:
                        chunk_x,chunk_mask = apply_attention_make_batch(x,mask,start_index,engine.config.tbtt_step,x_length,y_length)
                        chunk_x = chunk_x.to(device,non_blocking=engine.config.multi_gpu)
                        chunk_mask = chunk_mask.to(device,non_blocking=engine.config.multi_gpu)

                        y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((chunk_x,chunk_mask),chunk_y)# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40
                        
                    else:
                        chunk_x,chunk_mask = apply_attention_make_batch(x,mask,start_index,engine.config.tbtt_step , x_length, y_length)
                        chunk_x = chunk_x.to(device,non_blocking=engine.config.multi_gpu)
                        chunk_mask = chunk_mask.to(device,non_blocking=engine.config.multi_gpu)
                        encoder_hidden = detach_hidden(encoder_hidden)
                        decoder_hidden = detach_hidden(decoder_hidden)
                        y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((chunk_x,chunk_mask),chunk_y,en_hidden = encoder_hidden,de_hidden = decoder_hidden)# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40
                    
                    attention_index = np.array(torch.argmax(mini_attention[:,-1,:],dim=1).tolist())
                    chunk_index = chunk_index + engine.config.tbtt_step

                    loss = engine.crit(# pad만으로 구성돼서?
                        y_hat.contiguous().view(-1,y_hat.size(-1)),
                        chunk_y_label.contiguous().view(-1)
                    )

                    '''print('chunk_x',chunk_x.shape)
                    print('y_hat',y_hat.contiguous().view(-1,y_hat.size(-1)).shape)
                    print('y',chunk_y_label.contiguous().view(-1).shape)'''

                    soft_mask = guided_attentions(mini_attention.shape,engine.config.W)
                    soft_mask = torch.from_numpy(soft_mask).to(device)
                    attn_loss = -(soft_mask * mini_attention).mean() #sum or mean?
                    #if not torch.isnan(attn_loss):
                    loss = loss + attn_loss
                    #|y_hat| = (batch_size,len  gth,ouput_size)
                    '''print('soft_mask',torch.isnan(soft_mask).any())
                    print('mini_attention',torch.isnan(mini_attention).any())
                    print('attn_loss',attn_loss)
                    print('chunk_x',torch.isnan(chunk_x).any())
                    print('y_hat',torch.isnan(y_hat).any())
                    print('chunk_y_label',torch.isnan(chunk_y_label).any())'''

                    if engine.config.gpu_id >=0 or engine.engine.config.multi_gpu:
                        #print(1)
                        engine.scaler.scale(loss).backward()
                        engine.scaler.step(engine.optimizer)
                        engine.scaler.update()
                    else:
                        loss.backward()
                        engine.optimizer.step()


                    loss_list.append(loss.item())

                    '''torch_utils.clip_grad_norm_(
                        engine.model.parameters(),
                        engine.config.max_gr_norm,
                        #norm_type=2,
                    )'''#gradient clipping          

                    del chunk_y, chunk_y_label, chunk_x, chunk_mask, y_hat, mini_attention,loss
                        
        word_count = int(mini_batch_tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters())) #모델의 복잡도 학습됨에 따라 커져야함
        g_norm = float(get_grad_norm(engine.model.parameters()))    #클수록 뭔가 배우는게 변하는게 많다 (학습의 안정성)


        #if engine.config.use_noam_decay and engine.lr_scheduler is not None:
        #    engine.lr_scheduler.step()
        #print('loss_list',loss_list)
        loss = float((sum(loss_list)/len(loss_list))/word_count)
        ppl = np.exp(loss)   

        #print('train loss',loss)
        return {
            'loss': loss,
            'ppl': ppl,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
            '|p_param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        with torch.no_grad():
            device = next(engine.model.parameters()).device
            x,mask,x_length = mini_batch[0][0],mini_batch[0][1],mini_batch[0][2] #tensor,mask,length
            mini_batch_tgt = (mini_batch[1][0],mini_batch[1][1])
            y_length = mini_batch_tgt[1] 
            
            y = mini_batch_tgt[0][:,1:]  #<BOS> 제외 정답문장 1번 단어부터 비교
            #|x| = (batch_size,128,length)
            #|y| = (batch_size,length)

            encoder_hidden,decoder_hidden = None,None 
            loss_list = []
            chunk_index = 0
            start_index = np.zeros((x.size(0),), dtype=int)
            attention_index = 0
            input_y = mini_batch_tgt[0][:,:-1]
            with autocast():
                while chunk_index < engine.max_target_ratio * np.mean(y_length.tolist()):      
                    engine.model.eval()

                    chunk_y = input_y[:,chunk_index:chunk_index + engine.config.tbtt_step].to(device)
                    chunk_y_label = y[:,chunk_index:chunk_index + engine.config.tbtt_step].to(device)
                    
                    chunk_length = []
                    start_index = start_index + attention_index
                    
                    #print('start_index',start_index)
                    #print('attention_index',attention_index)
                        
                    #print('chunk_x:',chunk_x.shape)
                    if encoder_hidden is None:
                        chunk_x,chunk_mask = apply_attention_make_batch(x,mask,start_index,engine.config.tbtt_step,x_length,y_length)
                        chunk_x = chunk_x.to(device)
                        chunk_mask = chunk_mask.to(device)

                        y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((chunk_x,chunk_mask),chunk_y)# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40

                    else:
                        chunk_x,chunk_mask = apply_attention_make_batch(x,mask,start_index,engine.config.tbtt_step,x_length,y_length)
                        chunk_x = chunk_x.to(device)
                        chunk_mask = chunk_mask.to(device)

                        encoder_hidden = detach_hidden(encoder_hidden)
                        decoder_hidden = detach_hidden(decoder_hidden)
                        y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((chunk_x,chunk_mask),chunk_y,en_hidden = encoder_hidden,de_hidden = decoder_hidden)# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40
                    
                    attention_index = np.array(torch.argmax(mini_attention[:,-1,:],dim=1).tolist())
                    chunk_index = chunk_index + engine.config.tbtt_step

                    loss = engine.crit(
                        y_hat.contiguous().view(-1,y_hat.size(-1)),
                        chunk_y_label.contiguous().view(-1)
                    )

                    soft_mask = guided_attentions(mini_attention.shape,engine.config.W)
                    soft_mask = torch.from_numpy(soft_mask).to(device)
                    attn_loss = -(soft_mask * mini_attention).mean() #sum or mean?
                    loss = loss + attn_loss
                    #|y_hat| = (batch_size,length,ouput_size)
                    
                    
                    loss_list.append(loss.item())

                    del chunk_y, chunk_y_label, chunk_x, chunk_mask, y_hat, mini_attention,loss

        word_count = int(mini_batch_tgt[1].sum())
        loss = float((sum(loss_list)/len(loss_list))/word_count)
        ppl = np.exp(loss)   
        
        #print(loss)
        #print('target_ratio',engine.max_target_ratio)
        #print('epoch',engine.state.epoch)
        
        return {
            'loss': loss,
            'ppl': ppl
        }

    @staticmethod
    def test(engine,mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device 
            mini_batch_src = (mini_batch[0][0].to(device),mini_batch[0][1].to(device))
            mini_batch_tgt = (mini_batch[1][0].to(device),mini_batch[1][1])
            
            x, y = mini_batch_src, mini_batch_tgt[0][:,1:]
            #print(x[0].size())
            #|x| = (batch_size,length)
            #|y| = (batch_size,length)
            
            with autocast():
                y_hat,mini_attention,_,_ = engine.model(mini_batch_src,mini_batch_tgt[0][:,:-1])# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40

                #|y_hat| = (batch_size,n_class)
                
                loss = engine.crit(
                    y_hat.contiguous().view(-1,y_hat.size(-1)),
                    y.contiguous().view(-1),
                )
            
                soft_mask = guided_attentions(mini_attention.shape,engine.config.W)
                soft_mask = torch.from_numpy(soft_mask).to(device)
                attn_loss = -(soft_mask * mini_attention).mean() #sum or mean?
                loss = loss + attn_loss
                
        word_count = int(mini_batch_tgt[1].sum())
        loss = float(loss/word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl
        }

    @staticmethod
    def attach(train_engine, validation_engine, verbose = VERBOSE_BATCH_WISE): 
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform = lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )

        training_metric_name = ['loss', 'ppl', '|p_param|', '|g_param|']

        for metric_name in training_metric_name:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(train_engine, training_metric_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |p_params| = {:.2e} |g_param| = {:.2e} loss = {:.4e} ppl = {:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|p_param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['ppl'],
                ))

        validation_metrics_name = ['loss','ppl']

        for metrics in validation_metrics_name:
            attach_running_average(validation_engine, metrics)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(validation_engine, validation_metrics_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss = {:.4e} ppl = {:.4f} best_loss = {:.4e}'.format(             
                    engine.state.metrics['loss'],
                    engine.state.metrics['ppl'],
                    engine.best_loss
                ))

    @staticmethod
    def resume_training(engine,resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)
        
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def load_model_for_val(engin,train_engin):
        engein.model = train_engin.model
        print('loded')
        
    @staticmethod
    def save_model(engine, train_engine, config,vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        model_fn =  config.model_fn.split('.')
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (
                                        avg_train_loss,
                                        np.exp(avg_train_loss)
                                        ),
                                    '%.2f-%.2f' % (
                                        avg_valid_loss,
                                        np.exp(avg_valid_loss)
                                    )] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model':engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'vocab' : vocab,
            },model_fn
            )




class SingleTrainer():
    def __init__(self,target_engine_class,config):
        self.target_engine_class = target_engine_class
        self.config = config
        self.tb_logger = TensorboardLogger(log_dir = config.log_dir)
        super().__init__()

    def train(
        self, 
        model,crit,optimizer,train_loader,valid_loader,vocab,n_epochs,
        lr_scheduler = None
    ):
        #print(local_rank)
        self.train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model, crit, optimizer,lr_scheduler, self.config
        )

        self.valid_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model, crit, optimizer=None,lr_scheduler = None,
            config = self.config
        )

        self.tb_logger.attach_output_handler(
            self.train_engine,
            event_name =Events.EPOCH_COMPLETED,
            tag="training",
            metric_names = "all"
        )

        self.tb_logger.attach_output_handler(
            self.valid_engine,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names = "all",
            global_step_transform=global_step_from_engine(self.train_engine)
        )

        self.target_engine_class.attach(
            self.train_engine,
            self.valid_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, valid_engine, valid_loader):
            engine.max_target_ratio = engine.max_target_ratio + 0.02
            valid_engine.run(valid_loader, max_epochs =1)

        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            run_validation, #func
            self.valid_engine, valid_loader #args
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_STARTED,
            self.target_engine_class.load_model_for_val,
            self.train_engine
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            self.target_engine_class.check_best #func
        )

        
        self.train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            self.target_engine_class.save_model, # func
            self.train_engine, self.config,
            vocab #args
        )

        self.train_engine.run(
            train_loader,
            max_epochs = self.config.n_epochs
        )

        return model

    def test(self,test_loader):
        print('--------------train-------------------')
        self.valid_engine.run(
            test_loader,
            max_epochs=1
        )
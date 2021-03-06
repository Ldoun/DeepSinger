from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from  torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from alignment.utils import *
from ignite.contrib.handlers.tensorboard_logger import *

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from multiprocessing import Pool 

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
        self.max_target_ratio = self.config.max_ratio
        super().__init__(func)

        self.best_loss =np.inf
        self.best_model = None
        self.scaler = GradScaler(init_scale = config.init_scale)
        self.mini_attention = None
        self.cnt = 0

    @staticmethod
    def train(engine, mini_batch):
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

        total_acc, total_count = 0, 0
        encoder_hidden,decoder_hidden = None,None 
        loss_list = []
        attention_loss_list = []
        chunk_index = np.zeros((x.size(0),), dtype=int)
        start_index = np.zeros((x.size(0),), dtype=int)
        input_y = mini_batch_tgt[0][:,:-1]
        counter = 0
        pool = Pool(processes=20)
        
        #print(engine.max_target_ratio)
        
        while counter  < np.clip(engine.max_target_ratio,0,1) * (max(y_length.tolist()) -1):      
            engine.model.train()
            engine.optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                with autocast(engine.config.use_autocast):
                    chunk_y,chunk_y_label = y_make_batch(input_y,y,chunk_index,engine.config.tbtt_step)
                    chunk_y = chunk_y.view(input_y.size(0),-1).to(device)
                    chunk_y_label = chunk_y_label.view(input_y.size(0),-1).to(device)
                    
                    #start_index = start_index + attention_index
                    
                    #print('chunk_x:',chunk_x.shape)
                    if encoder_hidden is None:
                        chunk_x,chunk_mask = apply_attention_make_batch(x,mask,start_index,engine.config.tbtt_step,x_length,y_length)
                        chunk_x = chunk_x.to(device)
                        chunk_mask = chunk_mask.to(device)

                        y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((chunk_x,chunk_mask),(chunk_y))# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40
                        
                    else:
                        chunk_x,chunk_mask = apply_attention_make_batch(x,mask,start_index,engine.config.tbtt_step , x_length, y_length)
                        chunk_x = chunk_x.to(device)
                        chunk_mask = chunk_mask.to(device)
                        encoder_hidden = detach_hidden(encoder_hidden)
                        decoder_hidden = detach_hidden(decoder_hidden)
                        y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((chunk_x,chunk_mask),(chunk_y),en_hidden = encoder_hidden,de_hidden = decoder_hidden)# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40
                    

                    loss = engine.crit(# pad만으로 구성돼서?
                        y_hat.contiguous().view(-1,y_hat.size(-1)),
                        chunk_y_label.contiguous().view(-1)
                    )

                    total_acc += (y_hat.argmax(-1).view(-1) == chunk_y_label.view(-1)).sum().item()
                    total_count += chunk_y_label.size(0) * chunk_y_label.size(1)


                    soft_mask = guided_attentions(mini_attention.shape,engine.config.W)
                    soft_mask = torch.from_numpy(soft_mask).to(device)
                    attn_loss = -(soft_mask * mini_attention).mean() #sum or mean?
                    
                    loss = loss + attn_loss
                    attention_loss_list.append(float(attn_loss))

                    is_retrain = pool.map(get_next_index,mini_attention.clone().detach().cpu().numpy())
                    for i,retrain in enumerate(is_retrain):
                        if retrain == False:
                            continue
                        start_index[i] = start_index[i] + retrain
                        chunk_index[i] = chunk_index[i] + engine.config.tbtt_step

                    counter += engine.config.tbtt_step

            if (engine.config.gpu_id >=0 or engine.config.multi_gpu) and engine.config.use_autocast:
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

            del chunk_y, chunk_y_label, chunk_x, chunk_mask, y_hat,loss
                
        p_norm = float(get_parameter_norm(engine.model.parameters())) #모델의 복잡도 학습됨에 따라 커져야함
        g_norm = float(get_grad_norm(engine.model.parameters()))    #클수록 뭔가 배우는게 변하는게 많다 (학습의 안정성)

        if engine.lr_scheduler is not None:
            engine.lr_scheduler.step()

        engine.mini_attention = mini_attention[0,:,:x_length[0]].detach().cpu().numpy()
        engine.filename = mini_batch[2][0]
        engine.cnt += 1
        #if engine.config.use_noam_decay and engine.lr_scheduler is not None:
        #    engine.lr_scheduler.step()
        #print('loss_list',loss_list)
        loss = float(sum(loss_list)/len(loss_list))
        ppl = np.exp(loss)   
        attention_loss = float(sum(attention_loss_list)/len(attention_loss_list))

        #print('train loss',loss)
        return {
            'loss': loss,
            'acc' : total_acc/total_count,
            'ppl': ppl,
            'attention':attention_loss,
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

            total_acc, total_count = 0, 0
            encoder_hidden,decoder_hidden = None,None 
            loss_list = []
            chunk_index = 0
            start_index = np.zeros((x.size(0),), dtype=int)
            input_y = mini_batch_tgt[0][:,:-1]
            with autocast(engine.config.use_autocast):
                engine.model.eval()
                y = y.to(device)
                x = x.to(device)
                mask = mask.to(device)

                y_hat,mini_attention,encoder_hidden,decoder_hidden = engine.model((x,mask),mini_batch_tgt[0][:,:-1].to(device))# pad token? need fixing https://github.com/kh-kim/simple-nmt/issues/40

                loss = engine.crit(
                    y_hat.contiguous().view(-1,y_hat.size(-1)),
                    y.contiguous().view(-1)
                )

                total_acc += (y_hat.argmax(-1).view(-1) == y.view(-1)).sum().item()
                total_count += y.size(0) * y.size(1)

                soft_mask = guided_attentions(mini_attention.shape,engine.config.W)
                soft_mask = torch.from_numpy(soft_mask).to(device)
                attn_loss = -(soft_mask * mini_attention).mean() #sum or mean?
                #loss = loss + attn_loss
                #|y_hat| = (batch_size,length,ouput_size)

        word_count = int(mini_batch_tgt[1].sum())
        loss = float(loss)
        attention_loss = float(attn_loss)
        ppl = np.exp(loss)  

        engine.mini_attention = mini_attention[0,:,:x_length[0]].detach().cpu().numpy()
        engine.filename = mini_batch[2][0]

        return {
            'loss': loss,
            'acc' : total_acc/total_count,
            'ppl': ppl,
            'attention':attention_loss
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

        training_metric_name = ['loss','acc', 'ppl', 'attention', '|p_param|', '|g_param|']

        for metric_name in training_metric_name:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(train_engine, training_metric_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |p_params| = {:.2e} |g_param| = {:.2e} loss = {:.4e} acc = {:.4e} ppl = {:.4f} attention = {:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|p_param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['acc'],
                    engine.state.metrics['ppl'],
                    engine.state.metrics['attention']
                ))
                if engine.config.nohup:
                    print(' end \n')

        validation_metrics_name = ['loss','ppl','attention','acc']

        for metrics in validation_metrics_name:
            attach_running_average(validation_engine, metrics)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(validation_engine, validation_metrics_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss = {:.4e} acc = {:.4e} ppl = {:.4f} attention = {:.4f} best_loss = {:.4e}'.format(             
                    engine.state.metrics['loss'],
                    engine.state.metrics['acc'],
                    engine.state.metrics['ppl'],
                    engine.state.metrics['attention'],
                    engine.best_loss
                ))
                if engine.config.nohup:
                    print(' end \n')

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
    def log_attention_map(engine,train_engine,writer):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        im = ax.imshow( 
            engine.mini_attention,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        
        plt.ylabel('Decoder timestep')
        plt.xlabel('Encoder timestep')
        plt.title(engine.filename)
        plt.tight_layout()

        writer.add_figure('attention allignment', fig,train_engine.state.epoch)

    @staticmethod
    def training_log_attention_map(train_engine,writer):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        im = ax.imshow( 
            train_engine.mini_attention,
            aspect='auto',
            origin='lower',
            interpolation='none')
        fig.colorbar(im, ax=ax)
        
        plt.ylabel('Decoder timestep')
        plt.xlabel('Encoder timestep')
        plt.title(train_engine.filename)
        plt.tight_layout()

        writer.add_figure('attention allignment(train)', fig, train_engine.cnt)

        

    @staticmethod
    def save_model(engine, train_engine, config):
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

        if config.multi_gpu:
            model_data = engine.model.module.state_dict()
        else:
            model_data = engine.model.state_dict()

        torch.save(
            {
                'model':model_data,
                'opt': train_engine.optimizer.state_dict(),
                'scaler': train_engine.scaler.state_dict(),
                'config': config
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
        model,crit,optimizer,train_loader,valid_loader,n_epochs,
        lr_scheduler = None, scaler_weight = None
    ):
        #print(local_rank)
        self.train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model, crit, optimizer,lr_scheduler, self.config
        )

        if scaler_weight is not None:
            self.train_engine.scaler.load_state_dict(scaler_weight)

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

        self.train_engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            self.target_engine_class.training_log_attention_map,
            self.train_engine, self.tb_logger.writer
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
            self.train_engine, self.config
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            self.target_engine_class.log_attention_map, # func
            self.valid_engine,self.train_engine, self.tb_logger.writer
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
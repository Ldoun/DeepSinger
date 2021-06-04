import argparse 
import pprint
import torch 
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as custom_optim
from torch.utils.data import DataLoader, random_split

from alignment.trainer import SingleTrainer,MaximumLikelihoodEstimationEngine
from alignment.dataloader import LJSpeechDataset,RandomBucketBatchSampler,TextAudioCollate
from alignment.Tokenizer import tokenizer

from alignment.model.lyrics_alignment import alignment_model


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )

    p.add_argument(
        '--tsv',
        required=not is_continue,
        help='Training set file name'
    )
    
    p.add_argument(
        '--music_dir',
        required=not is_continue ,
        help='music folder path'
    )

    p.add_argument(
        '--train_size',
        type=int,
        default=2000,
        help='length of train set'
    )
    
    p.add_argument(
        '--valid_size',
        type=int,
        default=838,
        help='length of valid set'
    )

    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--dropout',
        type=float,
        default=.1,
        help='Dropout rate. Default=%(default)s'
    )

    p.add_argument(
        '--tbtt_step',
        type=int,
        default=40,
        help='tbtt_step. Default=%(default)s'
    )

    p.add_argument(
        '--W',
        type=int,
        default=55,
        help='W. Default=%(default)s'
    )

    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )

    p.add_argument(
        '--en_hs',
        type=int,
        default=512,
        help='encoder Hidden size'
    )
    
    p.add_argument(
        '--de_hs',
        type=int,
        default=1024,
        help='decoder Hidden size'
    )

    p.add_argument(
        '--attention_dim',
        type=int,
        default=256,
        help='attention dim size'
    )

    p.add_argument(
        '--location_feature_dim',
        type=int,
        default=128,
        help='location_feature dim size'
    )

    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )

    p.add_argument(
        '--multi_gpu',
        action='store_true',
        help='multi-gpu',
    )

    p.add_argument(
        '--log_dir',
        type=str,
        default='../tensorboard'
    )

    config = p.parse_args()

    return config

def get_model(input_size, output_size, config):
    model = alignment_model(
            input_size,
            output_size,
            config.word_vec_size,  
            config.en_hs,
            config.de_hs,
            config.attention_dim,
            config.location_feature_dim,
            config.dropout 
        )

    return model

def get_crit(output_size, pad_index):
    # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
    # Thus, set a weight for PAD to zero.
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    # Instead of using Cross-Entropy loss,
    # we can use Negative Log-Likelihood(NLL) loss with log-probability.
    crit = nn.CrossEntropyLoss(
        weight=loss_weight,
        reduction='mean'
    )

    return crit

def get_optimizer(model, config):
    if config.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=config.lr,weight_decay=1e-6)

    return optimizer

def main(config, model_weight=None, opt_weight=None, vocab = None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    tok = tokenizer(vocab)
    if vocab is None:
        tok.set_vocab(config.tsv)

    dataset = LJSpeechDataset(config.music_dir,config.tsv,tok = tok )

    train_dataset,valid_dataset = random_split(dataset,[config.train_size,config.valid_size],generator=torch.Generator().manual_seed(42))
    
    train_batch_sampler = RandomBucketBatchSampler(train_dataset, batch_size=config.batch_size, drop_last=False)
    valid_batch_sampler = RandomBucketBatchSampler(valid_dataset, batch_size=config.batch_size//2, drop_last=False)
    
    collate_fn = TextAudioCollate()

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler,collate_fn=collate_fn)

    #print(tok.vocab)
    #print('-' * 80)

    input_size, output_size = 128, len(tok.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, tok.pad)

    if model_weight is not None:
        model.load_state_dict(model_weight)

    # Pass models to GPU device if it is necessary.

    if config.multi_gpu:
        model = nn.DataParallel(model)
        model.cuda()
        crit.cuda()


    if config.gpu_id >= 0 and not config.multi_gpu:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)
  

    optimizer = get_optimizer(model, config)

    if opt_weight is not None and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    # Start training. This function maybe equivalant to 'fit' function in Keras.
    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)

    #mle_trainer.tb_logger.writer.add_graph(model=model,input_to_model=torch.randn(128,1,3000).to(device),verbose=True)

    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        vocab=tok.vocab,
        n_epochs=config.n_epochs,
    )

    mle_trainer.tb_logger.close()

if __name__ == '__main__':
    config = define_argparser()
    main(config)    
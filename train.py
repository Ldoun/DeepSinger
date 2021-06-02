import argparse 
import pprint
import torch 
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as custom_optim
from torch.utils.data import DataLoader

from alignment.trainer import SingleTrainer,MaximumLikelihoodEstimationEngine
from alignment.dataloader import LJSpeechDataset,RandomBucketBatchSampler,TextAudioCollate
import alignment.Tokenizer as tokenizer

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
        '--train',
        required=not is_continue,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--test',
        required=not is_continue,
        help='test set file name except the extention. (ex: test.en --> test)'
    )
    
    p.add_argument(
        '--music_dir',
        required=not is_continue ,
        help='music folder path'
    )

    p.add_argument(
        '--tsv',
        required=not is_continue ,
        help = 'tsv file path'
    )

    p.add_argument(
        '--vocab_f',
        required=not is_continue ,
        help = 'vocab pickle file path'
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
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )

    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )

    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )

    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
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
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )

    p.add_argument(
        '--log_dir',
        type=str,
        required=False,
        default='tensorboard_logs'
    )

    config = p.parse_args()

    return config

def get_model(input_size, output_size, config):
    model = alignment_model(
            input_size,
            config.word_vec_size,           # Word embedding vector size
            config.hidden_size,             # LSTM's hidden vector size
            output_size,
            n_layers=config.n_layers,       # number of layers in LSTM
            dropout_p=config.dropout        # dropout-rate in LSTM
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
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
        else: # case of rnn based seq2seq.
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    return optimizer

def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    tok = tokenizer(config.vocab_f)
    dataset = LJSpeechDataset(config.music_dir,config.tsv,tok = tok)
    batch_sampler = RandomBucketBatchSampler(dataset, batch_size=5, drop_last=False)
    collate_fn = TextAudioCollate()
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                            collate_fn=collate_fn)

    tok.save(config.vocab_f)

    loader = DataLoader(
        config.train,                           # Train file name except extention, which is language.
        config.valid,                           # Validation file name except extension.
        (config.lang[:2], config.lang[-3:]),    # Source and target language.
        batch_size=config.batch_size,
        device=-1,                              # Lazy loading
        max_length=config.max_length,           # Loger sequence will be excluded.
        dsl=False,                              # Turn-off Dual-supervised Learning mode.
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    if model_weight is not None:
        model.load_state_dict(model_weight)

    # Pass models to GPU device if it is necessary.
    if config.gpu_id >= 0:
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
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
    )

    mle_trainer.tb_logger.close()


if __name__ == '__main__':
    config = define_argparser()
    main(config)
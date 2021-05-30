import torch
import torchaudio
from torchtext import data,datasets
from torch.utils.data import Dataset, DataLoader
import os
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import SequentialSampler



class MelSongDataset(Dataset):
    def __init__(self,tsv_file,root_dir):
        super().__init__()
        self.root_dir = root_dir

        self.tgt = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            #fix_length=fix_length,
            init_token='<SOS>',
            eos_token='<EOS>'
        )

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        pass

    def get_mel(self,file_name):
        audio = torchaudio.load_wav(os.path.join(self.root_dir,file_name))
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050,n_fft=1024,hop_length=256)(audio)

        return mel.squeeze(0)

    
class RandomBucketBatchSampler(object):
    """Yields of mini-batch of indices, sequential within the batch, random between batches.
    
    I.e. it works like bucket, but it also supports random between batches.
    Helpful for minimizing padding while retaining randomness with variable length inputs.
    Args:
        data_source (Dataset): dataset to sample from.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, data_source, batch_size, drop_last):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                                "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                                "drop_last={}".format(drop_last))
        self.sampler = SequentialSampler(data_source) # impl sequential within the batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches() # impl random between batches
        
    def __init__(self, data_source, batch_size, drop_last):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                                "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                                "drop_last={}".format(drop_last))
        self.sampler = SequentialSampler(data_source) # impl sequential within the batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches() # impl random between batches

    def _make_batches(self):
        indices = [i for i in self.sampler]
        batches = [indices[i:i+self.batch_size]
                    for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and len(self.sampler) % self.batch_size > 0:
            random_indices = torch.randperm(len(batches)-1).tolist() + [len(batches)-1]
        else:
            random_indices = torch.randperm(len(batches)).tolist()
        return [batches[i] for i in random_indices]

    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)



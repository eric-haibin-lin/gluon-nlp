import torch
import mxnet as mx
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

def get_hd5_loader(path, rank, bs, max_pred_length, is_train):
    hd5_dir = path
    keyword = 'training' if is_train else 'test'
    hd5_files = [os.path.join(hd5_dir, f) for f in os.listdir(hd5_dir) if
             os.path.isfile(os.path.join(hd5_dir, f)) and keyword in f]
    hd5_files.sort()
    hd5_num_files = len(hd5_files)
    assert rank < len(hd5_files), (rank, len(hd5_files), path)
    hd5_data_file = hd5_files[rank]
    
    class hd5_pretraining_dataset(Dataset):
    
        def __init__(self, input_file, max_pred_length):
            self.input_file = input_file
            self.max_pred_length = max_pred_length
            f = h5py.File(input_file, "r")
            keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                    'next_sentence_labels']
            self.inputs = [np.asarray(f[key][:]) for key in keys]
            f.close()
    
        def __len__(self):
            return len(self.inputs[0])
    
        def __getitem__(self, index):
            [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
                torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                    np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]
    
            masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
            index = self.max_pred_length
            # store number of  masked tokens in index
            padded_mask_indices = (masked_lm_positions == 0).nonzero()
            if len(padded_mask_indices) != 0:
                index = padded_mask_indices[0].item()
            masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
    
            return [input_ids, segment_ids, input_mask,
                    masked_lm_labels, next_sentence_labels]
    
    # args.max_predictions_per_seq)
    hd5_train_data = hd5_pretraining_dataset(hd5_data_file, max_pred_length)
    hd5_train_sampler = RandomSampler(hd5_train_data) if is_train else SequentialSampler(hd5_train_data)
    hd5_train_dataloader = DataLoader(hd5_train_data, sampler=hd5_train_sampler,
                                  batch_size=bs, num_workers=4,
                                  pin_memory=True)
    
    return hd5_train_dataloader

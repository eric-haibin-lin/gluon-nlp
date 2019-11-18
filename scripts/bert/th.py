import torch
import mxnet as mx
import os
import h5py
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

def get_hd5_loader(path, rank, bs, max_pred_length):
    hd5_dir = path # '/fsx/datasets/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/'
    hd5_files = [os.path.join(hd5_dir, f) for f in os.listdir(hd5_dir) if
             os.path.isfile(os.path.join(hd5_dir, f)) and 'training' in f]
    hd5_files.sort()
    hd5_num_files = len(hd5_files)
    hd5_rank = 0
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
    hd5_train_sampler = RandomSampler(hd5_train_data)
    hd5_train_dataloader = DataLoader(hd5_train_data, sampler=hd5_train_sampler,
                                  batch_size=bs, num_workers=4,
                                  pin_memory=True)
    
    return hd5_train_dataloader
"""
hd5_train_dataloader = get_hd5_loader(0, 32, 128)
hd5_train_iter = iter(hd5_train_dataloader)
hd5_max_pred_length = 20

for step, batch in enumerate(hd5_train_iter):
    batch_size = batch[0].shape[0]
    #my_batch = batch
    #my_input_ids, my_segment_ids, my_input_mask, my_masked_lm_labels, my_next_sentence_labels = my_batch
    #my_input_ids = my_input_ids.numpy()
    #my_segment_ids = my_segment_ids.numpy()
    #my_input_mask = my_input_mask.numpy()
    #my_masked_lm_labels = my_masked_lm_labels.numpy()
    #my_next_sentence_labels = my_next_sentence_labels.numpy()

    #nd_input_ids = mx.nd.array(my_input_ids, dtype=my_input_ids.dtype)
    #nd_segment_ids = mx.nd.array(my_segment_ids, dtype=my_segment_ids.dtype)
    #nd_valid_length = mx.nd.array(my_input_mask.sum(axis=1), dtype='float32')
    ## nd_masked_id =
    #nd_next_sentence_label = mx.nd.array(my_next_sentence_labels, dtype='float32')
    #np_masked_position = np.zeros((batch_size, max_pred_length))
    #np_masked_id = np.zeros((batch_size, max_pred_length))
    #np_masked_weight = np.zeros((batch_size, max_pred_length))
    #for i in range(batch_size):
    #    row = my_masked_lm_labels[i]
    #    idx = (row + 1).nonzero()[0]
    #    np_masked_id[i][:len(idx)] = row[idx]
    #    np_masked_position[i][:len(idx)] = idx
    #    np_masked_weight[i][:len(idx)] = 1
    #nd_masked_position = mx.nd.array(np_masked_position)
    #nd_masked_id = mx.nd.array(np_masked_id)
    #nd_masked_weight = mx.nd.array(np_masked_weight)
    #nd_batch = [nd_input_ids, nd_masked_id, nd_masked_position, nd_masked_weight, \
    #            nd_next_sentence_label, nd_segment_ids, nd_valid_length]
    assert batch_size == 32, (step, batch_size)
"""

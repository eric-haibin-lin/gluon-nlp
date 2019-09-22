"""
Pre-training Bidirectional Encoder Representations from Transformers
=========================================================================================
This example shows how to pre-train a BERT model with Gluon NLP Toolkit.
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import random
import warnings
import logging
import functools
import time
import argparse

import mxnet as mx
import gluonnlp as nlp
try:
    import horovod.mxnet as hvd
except ImportError:
    pass

from fp16_utils import FP16Trainer
from pretraining_utils import get_model_loss, get_pretrain_data_npz, get_dummy_dataloader
from pretraining_utils import split_and_load, log, evaluate
from pretraining_utils import save_parameters, save_states, profile
from pretraining_utils import get_pretrain_data_text, generate_dev_set

# parser
parser = argparse.ArgumentParser(description='BERT pretraining example.')
# logging and serialization
parser.add_argument('--ckpt_dir', type=str, default='./ckpt_dir',
                    help='Path to checkpoint directory')
parser.add_argument('--log_interval', type=int, default=250, help='Report interval')
parser.add_argument('--ckpt_interval', type=int, default=25000, help='Checkpoint interval')
# model
parser.add_argument('--pretrained', action='store_true',
                    help='Initialize the model with pretrained weights')
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'],
                    help='Model to pre-train.')
parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                             'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                             'wiki_cn_cased'],
                    help='The pre-defined dataset from which the vocabulary is created.')
# training
parser.add_argument('--data', type=str, default=None,
                    help='Path to training data file. File name with wildcard such as '
                         'dir/*.train is accepted.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
parser.add_argument('--accumulate', type=int, default=1,
                    help='Number of batches for gradient accumulation. '
                         'The effective batch size = batch_size * accumulate.')
parser.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
parser.add_argument('--start_step', type=int, default=0,
                    help='Start optimization step from the checkpoint.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.01,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--dtype', type=str, default='float16', help='data dtype')
# validation
parser.add_argument('--eval_interval', type=int, default=50000, help='Evaluation interval')
parser.add_argument('--batch_size_eval', type=int, default=8,
                    help='Batch size per GPU for evaluation.')
parser.add_argument('--data_eval', type=str, required=True,
                    help='Path to evaluation data file. File name with wildcard such as '
                         'dir/*.dev is accepted.')
parser.add_argument('--eval_use_npz', action='store_true',
                    help='Set to True if --data_eval provides npz files instead of raw text files')
# debugging
parser.add_argument('--dummy_data_len', type=int, default=None,
                    help='If provided, a data batch of target sequence length is '
                         'used. For benchmarking purpuse only.')
parser.add_argument('--verbose', action='store_true', help='verbose logging')
parser.add_argument('--profile', type=str, default=None,
                        help='output profiling result to the target file')
# data pre-processing
parser.add_argument('--num_buckets', type=int, default=1,
                    help='Number of buckets for variable length sequence sampling')
parser.add_argument('--raw', action='store_true',
                    help='If set, both training and dev samples are generated on-the-fly '
                         'from raw texts instead of pre-processed npz files. ')
parser.add_argument('--max_seq_length', type=int, default=512,
                    help='Maximum input sequence length. Effective only if --raw is set.')
parser.add_argument('--short_seq_prob', type=float, default=0,
                    help='The probability of producing sequences shorter than max_seq_length. '
                         'Effective only if --raw is set.')
parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                    help='Probability for masks. Effective only if --raw is set.')
parser.add_argument('--max_predictions_per_seq', type=int, default=80,
                    help='Maximum number of predictions per sequence. '
                         'Effective only if --raw is set.')
parser.add_argument('--cased', action='store_true',
                    help='Whether to tokenize with cased characters. '
                         'Effective only if --raw is set.')
parser.add_argument('--whole_word_mask', action='store_true',
                    help='Whether to use whole word masking rather than per-subword masking.'
                         'Effective only if --raw is set.')
parser.add_argument('--sentencepiece', default=None, type=str,
                    help='Path to the sentencepiece .model file for both tokenization and vocab. '
                         'Effective only if --raw is set.')
parser.add_argument('--num_data_workers', type=int, default=8,
                    help='Number of workers to pre-process data.')
# communication
parser.add_argument('--comm_backend', type=str, default='device',
                    choices=['horovod', 'dist_sync_device', 'device'],
                    help='Communication backend.')

args = parser.parse_args()

# logging
level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(level)
logging.info(args)
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'

class DataParallelBERT(nlp.utils.Parallelizable):
    """Data parallel BERT model.

    Parameters
    ----------
    model : Block
        The BERT model.
    """
    def __init__(self, model, trainer):
        self._model = model
        self._trainer = trainer

    def forward_backward(self, x):
        """forward backward implementation"""
        (input_id, masked_id, masked_position, masked_weight, \
         next_sentence_label, segment_id, valid_length) = x

        valid_length = valid_length.astype(args.dtype, copy=False)
        with mx.autograd.record():
            out = self._model(input_id, masked_id, masked_position, masked_weight,
                              next_sentence_label, segment_id, valid_length)
            classified, decoded, ls1, ls2 = out
            ls = ls1 + ls2
            ls = ls / args.accumulate
        if self._trainer:
            self._trainer.backward(ls)
        else:
            ls.backward()

        masked_id = masked_id.reshape(-1)
        valid_length = valid_length.astype('float32', copy=False)
        return next_sentence_label, classified, masked_id, decoded, \
               masked_weight, ls1, ls2, valid_length

def init_comm(backend):
    """Init communication backend"""
    # backend specific implementation
    if backend == 'horovod':
        try:
            import horovod.mxnet as hvd
        except ImportError:
            logging.info('horovod must be installed.')
            exit()
        hvd.init()
        store = None
        num_workers = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        is_master_node = rank == local_rank
        ctxs = [mx.gpu(local_rank)]
    else:
        # kvstore
        store = mx.kv.create(backend)
        num_workers = store.num_workers
        rank = store.rank
        local_rank = 0
        is_master_node = rank == local_rank
        ctxs = os.environ.get('NVIDIA_VISIBLE_DEVICES', '0')
        ctxs = [mx.gpu(int(ctx)) for ctx in ctxs.split(',')]
    return store, num_workers, rank, local_rank, is_master_node, ctxs

backend = args.comm_backend
store, num_workers, rank, local_rank, is_master_node, ctxs = init_comm(backend)

def train(data_train, data_eval, model):
    """Training function."""
    # backend specific implementation
    param_dict = model.bert.collect_params()
    if backend == 'horovod':
        hvd.broadcast_parameters(param_dict, root_rank=0)

    mlm_metric = nlp.metric.MaskedAccuracy()
    nsp_metric = nlp.metric.MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    logging.debug('Creating distributed trainer...')
    lr = args.lr
    optim_params = {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
    if args.dtype == 'float16':
        optim_params['multi_precision'] = True

    dynamic_loss_scale = args.dtype == 'float16'
    if dynamic_loss_scale:
        loss_scale_param = {'scale_window': 2000 / num_workers}
    else:
        loss_scale_param = None

    # backend specific implementation
    if backend == 'horovod':
        trainer = hvd.DistributedTrainer(param_dict, 'bertadam', optim_params)
    else:
        trainer = mx.gluon.Trainer(param_dict, 'bertadam', optim_params,
                                   update_on_kvstore=False)
    fp16_trainer = FP16Trainer(trainer, dynamic_loss_scale=dynamic_loss_scale,
                               loss_scaler_params=loss_scale_param)

    if args.start_step:
        state_path = os.path.join(args.ckpt_dir, '%07d.states.%02d'%(args.start_step, local_rank))
        logging.info('Loading trainer state from %s', state_path)
        nlp.utils.load_states(trainer, state_path)

    accumulate = args.accumulate
    num_train_steps = args.num_steps
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    params = [p for p in param_dict.values() if p.grad_req != 'null']

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    if accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    train_begin_time = time.time()
    begin_time = time.time()
    running_mlm_loss, running_nsp_loss = 0, 0
    running_num_tks = 0
    batch_num = 0
    step_num = args.start_step

    logging.debug('Training started')

    # create dummy data loader if needed
    if args.dummy_data_len:
        target_shape = (args.batch_size, args.dummy_data_len)
        data_train = get_dummy_dataloader(data_train, target_shape)

    parallel_model = DataParallelBERT(model, trainer=fp16_trainer)
    num_ctxes = len(ctxs)
    parallel = nlp.utils.Parallel(num_ctxes if num_ctxes > 1 else 0, parallel_model)

    while step_num < num_train_steps:
        for _, data_batch in enumerate(data_train):
            if step_num >= num_train_steps:
                break
            if batch_num % accumulate == 0:
                step_num += 1
                # if accumulate > 1, grad_req is set to 'add', and zero_grad is required
                if accumulate > 1:
                    param_dict.zero_grad()
                # update learning rate
                if step_num <= num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    offset = lr * step_num / num_train_steps
                    new_lr = lr - offset
                trainer.set_learning_rate(new_lr)
                if args.profile:
                    profile(step_num, 10, 14, profile_name=args.profile + str(rank))

            # load data
            data_list = list(split_and_load(data_batch, ctxs))

            ns_label_list, ns_pred_list = [], []
            mask_label_list, mask_pred_list, mask_weight_list = [], [], []

            with mx.autograd.record():
                num_data = len(data_list)
                for i in range(num_data):
                    parallel.put(data_list[i])
                for _ in range(num_data):
                    (next_sentence_label, classified, masked_id,
                     decoded, masked_weight, ls1, ls2, valid_length) = parallel.get()
                    ns_label_list.append(next_sentence_label)
                    ns_pred_list.append(classified)
                    mask_label_list.append(masked_id)
                    mask_pred_list.append(decoded)
                    mask_weight_list.append(masked_weight)
                    running_mlm_loss += ls1.as_in_context(mx.cpu()) / len(ctxs)
                    running_nsp_loss += ls2.as_in_context(mx.cpu()) / len(ctxs)
                    running_num_tks += valid_length.sum().as_in_context(mx.cpu())

            # update
            if (batch_num + 1) % accumulate == 0:
                fp16_trainer.step(1, max_norm=1.0 * num_workers)
            nsp_metric.update(ns_label_list, ns_pred_list)
            mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)

            # logging
            if (step_num + 1) % (args.log_interval) == 0 and (batch_num + 1) % accumulate == 0:
                log(begin_time, running_num_tks, running_mlm_loss / accumulate,
                    running_nsp_loss / accumulate, step_num, mlm_metric, nsp_metric,
                    trainer, args.log_interval)
                begin_time = time.time()
                running_mlm_loss = running_nsp_loss = running_num_tks = 0
                mlm_metric.reset_local()
                nsp_metric.reset_local()

            # saving checkpoints
            if (step_num + 1) % args.ckpt_interval == 0 and (batch_num + 1) % accumulate == 0:
                if is_master_node:
                    save_states(step_num, trainer, args.ckpt_dir, local_rank)
                    if local_rank == 0:
                        save_parameters(step_num, model.bert, args.ckpt_dir)
                if (step_num + 1) % args.eval_interval == 0 and data_eval:
                    # eval data is always based on a fixed npz file.
                    dataset_eval = get_pretrain_data_npz(data_eval, args.batch_size_eval,
                                                         1, False, 1, vocab)
                    evaluate(dataset_eval, model, ctxs, args.log_interval, args.dtype)

            batch_num += 1

    if is_master_node:
        save_states(step_num, trainer, args.ckpt_dir, local_rank)
        if local_rank == 0:
            save_parameters(step_num, model, args.ckpt_dir)
    mx.nd.waitall()
    train_end_time = time.time()
    logging.info('Train cost={:.1f}s'.format(train_end_time - train_begin_time))

if __name__ == '__main__':
    random_seed = random.randint(0, 1000)
    nlp.utils.mkdir(args.ckpt_dir)

    dataset_name, vocab = args.dataset_name, None
    if args.sentencepiece:
        logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
        if args.dataset_name:
            warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                          'The vocabulary will be loaded based on --sentencepiece')
            dataset_name = None
        vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)

    model, vocab = get_model_loss(ctxs, args.model, args.pretrained,
                                  dataset_name, vocab, args.dtype,
                                  ckpt_dir=args.ckpt_dir,
                                  start_step=args.start_step)
    logging.debug('Model created')
    data_eval = args.data_eval

    if args.raw:
        if args.sentencepiece:
            tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab,
                                                 lower=not args.cased)
        else:
            tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=not args.cased)

        cache_dir = os.path.join(args.ckpt_dir, 'data_eval_cache')
        cache_file = os.path.join(cache_dir, 'part-000.npz')
        nlp.utils.mkdir(cache_dir)

        # generate dev dataset from the raw text if needed
        if not args.eval_use_npz:
            data_eval = cache_file
            if not os.path.isfile(cache_file) and rank == 0:
                generate_dev_set(tokenizer, vocab, cache_file, args)

    logging.debug('Random seed set to %d', random_seed)
    mx.random.seed(random_seed)

    if args.data:
        if args.raw:
            get_dataset_fn = functools.partial(get_pretrain_data_text,
                                               max_seq_length=args.max_seq_length,
                                               short_seq_prob=args.short_seq_prob,
                                               masked_lm_prob=args.masked_lm_prob,
                                               max_predictions_per_seq=args.max_predictions_per_seq,
                                               whole_word_mask=args.whole_word_mask,
                                               tokenizer=tokenizer)
        else:
            get_dataset_fn = get_pretrain_data_npz

        num_parts = 1 if args.dummy_data_len else num_workers
        part_idx = 0 if args.dummy_data_len else rank
        data_train = get_dataset_fn(args.data, args.batch_size, 1, True,
                                    args.num_buckets, vocab,
                                    num_parts=num_parts, part_idx=part_idx,
                                    num_workers=args.num_data_workers)
        train(data_train, data_eval, model)
    if data_eval:
        # eval data is always based on a fixed npz file.
        dataset_eval = get_pretrain_data_npz(data_eval, args.batch_size_eval, 1,
                                             False, 1, vocab)
        evaluate(dataset_eval, model, ctxs, args.log_interval, args.dtype)

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

"""Utilities for pre-training."""
import time
import os
import logging
import random
import multiprocessing

import numpy as np
import mxnet as mx
import gluonnlp as nlp

from data.pretrain import BERTSamplerFn, BERTDataLoaderFn
from data.dataloader import SimpleDatasetFn, DatasetLoader
from create_pretraining_data import create_training_instances

import math
import warnings
import random
import numpy as np
from mxnet.gluon.data import Sampler

__all__ = ['get_model_loss', 'get_pretrain_data_npz', 'get_dummy_dataloader',
           'save_parameters', 'save_states', 'evaluate', 'split_and_load',
           'get_pretrain_data_text', 'generate_dev_set', 'profile']

def _masked_softmax(F, att_score, mask, dtype):
    """Ignore the masked elements when calculating the softmax

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    mask : Symbol or NDArray or None
        Shape (batch_size, query_length, memory_length)
    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        neg = -1e4
        att_score = F.where(mask, att_score, neg * F.ones_like(att_score))
        att_weights = F.softmax(att_score, axis=-1) * mask
    else:
        att_weights = F.softmax(att_score, axis=-1)
    return att_weights

class FP32LayerNorm(mx.gluon.nn.LayerNorm):
    """BERT style Layer Normalization.

    Epsilon is added inside the square root and set to 1e-12 by default.

    Inputs:
        - **data**: input tensor with arbitrary shape.
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, epsilon=1e-12, in_channels=0, prefix=None, params=None):
        super(FP32LayerNorm, self).__init__(epsilon=epsilon, in_channels=in_channels,
                                            prefix=prefix, params=params)
    def cast(self, dtype):
        logging.info("Using FP32 layernorm")

    def hybrid_forward(self, F, data, gamma, beta):
        """forward computation."""
        logging.info("Using FP32 layernorm")
        return F.LayerNorm(data.astype('float32'), gamma=gamma, beta=beta, axis=self._axis, eps=self._epsilon).astype('float16')

if int(os.environ.get('FP32_LN', False)):
    nlp.model.bert.BERTLayerNorm = FP32LayerNorm

if int(os.environ.get('SMALL_NEG', False)):
    print("Using small NEG")
    nlp.model.attention_cell._masked_softmax = _masked_softmax

def convert_pytorch_to_mxnet(args, data_batch, batch_size):
    max_pred_length = args.max_predictions_per_seq
    my_batch = data_batch
    if my_batch[0].shape[0] != batch_size:
        return None
    my_input_ids, my_segment_ids, my_input_mask, my_masked_lm_labels, my_next_sentence_labels = my_batch
    my_input_ids = my_input_ids.numpy()
    my_segment_ids = my_segment_ids.numpy()
    my_input_mask = my_input_mask.numpy()
    my_masked_lm_labels = my_masked_lm_labels.numpy()
    my_next_sentence_labels = my_next_sentence_labels.numpy()

    nd_input_ids = mx.nd.array(my_input_ids, dtype=my_input_ids.dtype)
    nd_segment_ids = mx.nd.array(my_segment_ids, dtype=my_segment_ids.dtype)
    nd_valid_length = mx.nd.array(my_input_mask.sum(axis=1), dtype='float32')
    # nd_masked_id =
    nd_next_sentence_label = mx.nd.array(my_next_sentence_labels, dtype='float32')
    np_masked_position = np.zeros((batch_size, max_pred_length))
    np_masked_id = np.zeros((batch_size, max_pred_length))
    np_masked_weight = np.zeros((batch_size, max_pred_length))
    for i in range(batch_size):
        row = my_masked_lm_labels[i]
        idx = (row + 1).nonzero()[0]
        np_masked_id[i][:len(idx)] = row[idx]
        np_masked_position[i][:len(idx)] = idx
        np_masked_weight[i][:len(idx)] = 1
    nd_masked_position = mx.nd.array(np_masked_position)
    nd_masked_id = mx.nd.array(np_masked_id)
    nd_masked_weight = mx.nd.array(np_masked_weight)
    data_batch = [nd_input_ids, nd_masked_id, nd_masked_position, nd_masked_weight, \
                nd_next_sentence_label, nd_segment_ids, nd_valid_length]
    return data_batch

def _fp32_masked_softmax(F, att_score, mask, dtype):
    """Ignore the masked elements when calculating the softmax

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    mask : Symbol or NDArray or None
        Shape (batch_size, query_length, memory_length)
    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    # Fill in the masked scores with a very small value
    logging.info("Using FP32 SoftMax")
    neg = -1e18
    mask = mask.astype('float32')
    att_score = att_score.astype('float32')
    att_score = F.where(mask, att_score, neg * F.ones_like(att_score))
    att_weights = F.softmax(att_score, axis=-1) * mask
    return att_weights.astype('float16')

if int(os.environ.get('FP32_SM', False)):
    nlp.model.attention_cell._masked_softmax = _fp32_masked_softmax

class ShuffleSplitSampler(Sampler):
    """Split the dataset into `num_parts` parts and randomly sample from the part
    with index `part_index`.

    The data is randomly shuffled at each iteration within each partition.

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Number of partitions which the data is split into
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0, seed=0):
        if length % num_parts != 0:
            warnings.warn('Length ({}) must be a multiple of the number of partitions ({}).'.format(length, num_parts))
        self._seed = seed
        self._state = np.random.RandomState(seed)
        self._indices = list(range(length))
        # Compute the length of each partition
        part_len = length // num_parts
        # Compute the start index for this partition
        self._start = part_len * part_index
        # Compute the end index for this partition
        self._end = self._start + part_len
        if part_index == num_parts - 1:
            self._end = length

    def __iter__(self):
        self._state.shuffle(self._indices)
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(self._indices[self._start:self._end])
        return iter(indices)

    def __len__(self):
        return self._end - self._start

class BERTEncoder2(nlp.model.transformer.BaseTransformerEncoder):
    """Structure of the BERT Encoder.
    """

    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='gelu', layer_norm_eps=None):
        logging.info('Use BERTEncoder2 with embed_dropout = 0, use_layer_norm_before_dropout=True')
        super(BERTEncoder2, self).__init__(attention_cell=attention_cell,
                                           num_layers=num_layers, units=units,
                                           hidden_size=hidden_size, max_length=max_length,
                                           num_heads=num_heads, scaled=scaled, dropout=dropout,
                                           use_residual=use_residual,
                                           output_attention=output_attention,
                                           output_all_encodings=output_all_encodings,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer,
                                           prefix=prefix, params=params,
                                           # extra configurations for BERT
                                           positional_weight='learned',
                                           use_bert_encoder=True,
                                           use_layer_norm_before_dropout=True,
                                           scale_embed=False,
                                           activation=activation,
                                           layer_norm_eps=layer_norm_eps)

if int(os.environ.get('FIX_BERT_ENCODER', False)):
    nlp.model.bert.BERTEncoder = BERTEncoder2
    nlp.model.bert.bert_24_1024_16_hparams['embed_dropout'] = 0.0

def _encode_sequence(self, inputs, token_types, valid_length=None):
    """Generate the representation given the input sequences.
    This is used for pre-training or fine-tuning a BERT model.
    """
    # embedding
    embedding = self.word_embed(inputs)
    if self._use_token_type_embed:
        type_embedding = self.token_type_embed(token_types)
        embedding = embedding + type_embedding
    embedding = embedding.transpose((1, 0, 2))
    # encoding
    outputs, additional_outputs = self.encoder(embedding, valid_length=valid_length)
    outputs = outputs.transpose((1, 0, 2))
    return outputs, additional_outputs

def _arange_like(self, F, inputs, axis):
    """Helper function to generate indices of a range"""
    if F == mx.ndarray:
        seq_len = inputs.shape[axis]
        arange = F.arange(seq_len, dtype=inputs.dtype, ctx=inputs.context)
    else:
        end = [1,1,1]
        end[axis] = None
        input_axis = inputs.slice(begin=(0, 0, 0), end=tuple(end)).reshape((-1))
        zeros = F.zeros_like(input_axis)
        arange = F.arange(start=0, repeat=1, step=1,
                          infer_range=True, dtype=self._dtype)
        arange = F.elemwise_add(arange, zeros)
    return arange

def _transformer_hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
    # pylint: disable=arguments-differ
    """Encode the inputs given the states and valid sequence length.
    """
    # XXX Temporary hack for hybridization as hybridblock does not support None inputs
    if isinstance(valid_length, list) and len(valid_length) == 0:
        valid_length = None
    if isinstance(states, list) and len(states) == 0:
        states = None

    steps = self._arange_like(F, inputs, axis=0)
    if valid_length is not None:
        ones = F.ones_like(steps)
        mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
                                  F.reshape(valid_length, shape=(-1, 1)))
        mask = F.broadcast_mul(F.expand_dims(mask, axis=1),
                               F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
        if states is None:
            states = [mask]
        else:
            states.append(mask)
    if self._scale_embed:
        # XXX: input.shape[-1] and self._units are expected to be the same
        inputs = inputs * math.sqrt(self._units)
    if states is None:
        states = [steps]
    else:
        states.append(steps)
    if states is not None:
        steps = states[-1]
        # positional encoding
        positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
        inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=1))
    if self._dropout:
        if self._use_layer_norm_before_dropout:
            inputs = self.layer_norm(inputs)
            inputs = self.dropout_layer(inputs)
        else:
            inputs = self.dropout_layer(inputs)
            inputs = self.layer_norm(inputs)
    else:
        inputs = self.layer_norm(inputs)
    outputs = inputs
    if valid_length is not None:
        mask = states[-2]
    else:
        mask = None
    all_encodings_outputs = []
    additional_outputs = []
    for cell in self.transformer_cells:
        outputs, attention_weights = cell(inputs, mask)
        inputs = outputs
        if self._output_all_encodings:
            if valid_length is not None:
                outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                         use_sequence_length=True, axis=0)
            all_encodings_outputs.append(outputs)

        if self._output_attention:
            additional_outputs.append(attention_weights)

    if valid_length is not None:
        outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                 use_sequence_length=True, axis=0)

    if self._output_all_encodings:
        return all_encodings_outputs, additional_outputs
    else:
        return outputs, additional_outputs


if int(os.environ.get('USE_SA', False)):
    print('USING self attention')
    nlp.model.bert.BERTModel._encode_sequence = _encode_sequence
    nlp.model.transformer.BaseTransformerEncoder._arange_like = _arange_like
    nlp.model.transformer.BaseTransformerEncoder.hybrid_forward = _transformer_hybrid_forward

class RepeatSplitSampler(nlp.data.SplitSampler):
    def __init__(self, length, num_parts=1, part_index=0, repeat=40):
        super(RepeatSplitSampler, self).__init__(length, num_parts=num_parts, part_index=part_index)
        self.repeat = repeat

    def __iter__(self):
        l = []
        for i in range(self.repeat):
            l.extend(list(super(RepeatSplitSampler, self).__iter__()))
        return iter(l)

def get_model_loss(ctx, model, pretrained, dataset_name, vocab, dtype,
                   ckpt_dir=None, start_step=None):
    """Get model for pre-training.

    Parameters
    ----------
    ctx : Context or list of Context
        Contexts to initialize model
    model : str
        The name of the model, 'bert_12_768_12' or 'bert_24_1024_16'.
    pretrained : bool
        Whether to use pre-trained model weights as initialization.
    dataset_name : str
        The name of the dataset, which is used to retrieve the corresponding vocabulary file
        when the vocab argument is not provided. Options include 'book_corpus_wiki_en_uncased',
        'book_corpus_wiki_en_cased', 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'wiki_cn_cased'.
    vocab : BERTVocab or None
        The vocabulary for the model. If not provided, The vocabulary will be constructed
        based on dataset_name.
    dtype : float
        Data type of the model for training.
    ckpt_dir : str
        The path to the checkpoint directory.
    start_step : int or None
        If provided, it loads the model from the corresponding checkpoint from the ckpt_dir.

    Returns
    -------
    BERTForPretrain : the model for pre-training.
    BERTVocab : the vocabulary.
    """
    # model
    if int(os.environ.get('NO_DROPOUT', False)):
        logging.info("disabling dropout")
        nlp.model.bert.bert_24_1024_16_hparams['dropout'] = 0.0
        nlp.model.bert.bert_24_1024_16_hparams['embed_dropout'] = 0.0
    model, vocabulary = nlp.model.get_model(model, dataset_name=dataset_name, vocab=vocab,
                                            pretrained=pretrained, ctx=ctx)

    if not pretrained:
        if int(os.environ.get('TRUNCATE_NORM', False)):
            logging.info('Using truncated norm initialization')
            model.initialize(init=nlp.initializer.TruncNorm(0.02), ctx=ctx)
        else:
            model.initialize(init=mx.init.Normal(0.02), ctx=ctx)

    if not int(os.environ.get('USE_AMP', False)):
        model.cast(dtype)

    load_again = False
    if ckpt_dir and start_step:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        try:
            nlp.utils.load_parameters(model, param_path, ctx=ctx, cast_dtype=True)
            logging.info('Loading step %d checkpoints from %s.', start_step, param_path)
        except AssertionError:
            load_again = True
    # losses
    nsp_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    if not int(os.environ.get('USE_AMP', False)):
        nsp_loss.hybridize(static_alloc=True, static_shape=True)
        mlm_loss.hybridize(static_alloc=True, static_shape=True)

    model = BERTForPretrain(model, nsp_loss, mlm_loss, len(vocabulary))
    if not int(os.environ.get('USE_AMP', False)):
        model.hybridize(static_alloc=True, static_shape=True)

    if load_again:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        nlp.utils.load_parameters(model, param_path, ctx=ctx, cast_dtype=True)
        logging.info('Loading step %d checkpoints from %s.', start_step, param_path)
    return model, vocabulary

class BERTPretrainDataset(mx.gluon.data.ArrayDataset):
    """Dataset for BERT pre-training.

    Each record contains the following numpy ndarrays: input_ids, masked_lm_ids,
    masked_lm_positions, masked_lm_weights, next_sentence_labels, segment_ids, valid_lengths.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    tokenizer : BERTTokenizer
        The BERTTokenizer
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    vocab : BERTVocab
        The BERTVocab
    num_workers : int
        The number of worker processes for dataset contruction.
    worker_pool : multiprocessing.Pool
        The worker process pool. Must be provided if num_workers > 1.
    """
    def __init__(self, filename, tokenizer, max_seq_length, short_seq_prob,
                 masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                 vocab, num_workers=1, worker_pool=None):
        logging.debug('start to load file %s ...', filename)
        dupe_factor = 1
        if not isinstance(filename, (list, tuple)):
            filename = [filename]
        instances = create_training_instances((filename, tokenizer, max_seq_length,
                                               short_seq_prob, masked_lm_prob,
                                               max_predictions_per_seq,
                                               whole_word_mask, vocab,
                                               dupe_factor, num_workers,
                                               worker_pool, None))
        super(BERTPretrainDataset, self).__init__(*instances)

def get_pretrain_data_text(data, batch_size, num_ctxes, shuffle,
                           num_buckets, vocab, tokenizer, max_seq_length, short_seq_prob,
                           masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                           num_parts=1, part_idx=0, num_workers=1):
    """Get a data iterator from raw text documents.

    Parameters
    ----------
    batch_size : int
        The batch size per GPU.
    num_ctxes : int
        The number of GPUs.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : BERTVocab
        The vocabulary.
    tokenizer : BERTTokenizer or BERTSPTokenizer
        The tokenizer.
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs.
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    num_parts : int
        The number of partitions for the dataset.
    part_idx : int
        The index of the partition to read.
    num_workers : int
        The number of worker processes for dataset contruction.
    """
    num_files = len(nlp.utils.glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of training text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.'%(num_parts, num_files, data)
    dataset_params = {'tokenizer': tokenizer, 'max_seq_length': max_seq_length,
                      'short_seq_prob': short_seq_prob, 'masked_lm_prob': masked_lm_prob,
                      'max_predictions_per_seq': max_predictions_per_seq, 'vocab':vocab,
                      'whole_word_mask': whole_word_mask}
    dataset_fn = SimpleDatasetFn(BERTPretrainDataset, dataset_params)
    sampler_fn = BERTSamplerFn(batch_size, shuffle, num_ctxes, num_buckets)
    dataloader_fn = BERTDataLoaderFn(num_ctxes, vocab)

    if int(os.environ.get('REPEAT_SAMPLER', False)):
        file_sampler_cls = RepeatSplitSampler
    else:
        file_sampler_cls = nlp.data.SplitSampler
    if int(os.environ.get('EVEN_SHUFFLE', False)):
        file_sampler_cls = ShuffleSplitSampler
    split_sampler = file_sampler_cls(num_files, num_parts=num_parts, part_index=part_idx)
    dataloader = DatasetLoader(data, split_sampler, dataset_fn, sampler_fn, dataloader_fn,
                               num_dataset_workers=num_workers)
    return dataloader


def get_pretrain_data_npz(data, batch_size, num_ctxes, shuffle, num_buckets,
                          vocab, num_parts=1, part_idx=0, num_workers=1):
    """Get a data iterator from pre-processed npz files.

    Parameters
    ----------
    batch_size : int
        The batch size per GPU.
    num_ctxes : int
        The number of GPUs.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : BERTVocab
        The vocabulary.
    num_parts : int
        The number of partitions for the dataset.
    part_idx : int
        The index of the partition to read.
    num_workers : int
        The number of worker processes for dataset contruction.
    """
    num_files = len(nlp.utils.glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of training text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.'%(num_parts, num_files, data)
    #split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts, part_index=part_idx)
    dataset_params = {'allow_pickle' : True}
    dataset_fn = SimpleDatasetFn(nlp.data.NumpyDataset, dataset_params)
    sampler_fn = BERTSamplerFn(batch_size, shuffle, num_ctxes, num_buckets)
    dataloader_fn = BERTDataLoaderFn(num_ctxes, vocab)

    split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts, part_index=part_idx)
    dataloader = DatasetLoader(data, split_sampler, dataset_fn, sampler_fn, dataloader_fn,
                               num_dataset_workers=num_workers)
    return dataloader


def get_dummy_dataloader(batch_size, seq_len, max_predict):
    """Return a dummy data loader which returns a fixed data batch of target shape"""
    class DummyIter():
        def __init__(self, batch):
            self._batch = batch

        def __iter__(self):
            while True:
                yield self._batch
    data_batch = ((mx.nd.ones((batch_size, seq_len)),
                   mx.nd.ones((batch_size, max_predict)),
                   mx.nd.ones((batch_size, max_predict)),
                   mx.nd.ones((batch_size, max_predict)),
                   mx.nd.ones((batch_size,)) * seq_len,
                   mx.nd.ones((batch_size, seq_len)),
                   mx.nd.ones((batch_size,)) * seq_len))
    return DummyIter(data_batch)


def save_parameters(step_num, model, ckpt_dir):
    """Save the model parameter, marked by step_num."""
    param_path = os.path.join(ckpt_dir, '%07d.params'%step_num)
    logging.info('[step %d] Saving model params to %s.', step_num, param_path)
    nlp.utils.save_parameters(model, param_path)

def save_states(step_num, trainer, ckpt_dir, local_rank=0):
    """Save the trainer states, marked by step_num."""
    trainer_path = os.path.join(ckpt_dir, '%07d.states.%02d'%(step_num, local_rank))
    logging.info('[step %d] Saving trainer states to %s.', step_num, trainer_path)
    nlp.utils.save_states(trainer, trainer_path)

def log_noacc(begin_time, running_num_tks, running_mlm_loss, running_nsp_loss, step_num,
              trainer, log_interval):
    """Log training progress."""
    end_time = time.time()
    duration = end_time - begin_time
    throughput = running_num_tks / duration / 1000.0
    running_mlm_loss = running_mlm_loss / log_interval
    running_nsp_loss = running_nsp_loss / log_interval
    lr = trainer.learning_rate if trainer else 0
    # pylint: disable=line-too-long
    logging.info('[step {}]\tmlm_loss={:7.5f}\tnsp_loss={:5.2f}\tthroughput={:.1f}K tks/s\tlr={:.7f} time={:.2f}, latency={:.1f} ms/batch'
                 .format(step_num, running_mlm_loss.asscalar(), 0,
                         throughput.asscalar(), lr, duration, duration*1000/log_interval))
    # pylint: enable=line-too-long

def log(begin_time, running_num_tks, running_mlm_loss, running_nsp_loss, step_num,
        mlm_metric, nsp_metric, trainer, log_interval):
    """Log training progress."""
    end_time = time.time()
    duration = end_time - begin_time
    throughput = running_num_tks / duration / 1000.0
    running_mlm_loss = running_mlm_loss / log_interval
    running_nsp_loss = running_nsp_loss / log_interval
    lr = trainer.learning_rate if trainer else 0
    # pylint: disable=line-too-long
    logging.info('[step {}]\tmlm_loss={:7.5f}\tmlm_acc={:4.2f}\tnsp_loss={:5.2f}\tnsp_acc={:5.2f}\tthroughput={:.1f}K tks/s\tlr={:.7f} time={:.2f}, latency={:.1f} ms/batch'
                 .format(step_num, running_mlm_loss.asscalar(), mlm_metric.get()[1] * 100, running_nsp_loss.asscalar(),
                         nsp_metric.get()[1] * 100, throughput.asscalar(), lr, duration, duration*1000/log_interval))
    # pylint: enable=line-too-long

def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [mx.gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
    return zip(*loaded_arrs)

class BERTForPretrain(mx.gluon.Block):
    """Model for pre-training MLM and NSP with BERT.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    mlm_loss : Loss or None
    nsp_loss : Loss or None
    vocab_size : int
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert, mlm_loss, nsp_loss, vocab_size, prefix=None, params=None):
        super(BERTForPretrain, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        self.mlm_loss = mlm_loss
        self.nsp_loss = nsp_loss
        self._vocab_size = vocab_size

    def forward(self, input_id, masked_id, masked_position, masked_weight,
                next_sentence_label=None, segment_id=None, valid_length=None):
        # pylint: disable=arguments-differ
        """Predict with BERT for MLM and NSP. """
        num_masks = masked_weight.sum()
        valid_length = valid_length.reshape(-1)
        masked_id = masked_id.reshape(-1)
        _, _, classified, decoded = self.bert(input_id, segment_id, valid_length, masked_position)
        decoded = decoded.reshape((-1, self._vocab_size))
        ls1 = self.mlm_loss(decoded.astype('float32', copy=False),
                            masked_id, masked_weight.reshape((-1, 1)))
        ls2 = self.nsp_loss(classified.astype('float32', copy=False), next_sentence_label)
        ls1 = ls1.sum() / (num_masks + 1e-7)
        ls2 = ls2.mean()
        return classified, decoded, ls1, ls2, num_masks

def evaluate(data_eval, model, ctx, args, batch_size_eval): #log_interval, dtype, rank, num_workers, args):
    """Evaluation function."""
    log_interval = args.log_interval
    dtype = args.dtype
    logging.info('Running evaluation ... ')
    mlm_metric = nlp.metric.MaskedAccuracy()
    nsp_metric = nlp.metric.MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    eval_begin_time = time.time()
    begin_time = time.time()
    step_num = 0
    running_mlm_loss = running_nsp_loss = 0
    total_mlm_loss = total_nsp_loss = 0
    running_num_tks = 0

    import horovod.mxnet as hvd
    if int(os.environ.get('HD5', False)):
        from th import get_hd5_loader
        logging.info('using HD5 dataset for eval: {}'.format(args.data_eval))
        data_eval = get_hd5_loader(args.data_eval, hvd.rank(), batch_size_eval, args.max_predictions_per_seq, False)

    for idx, data_batch in enumerate(data_eval):
        if int(os.environ.get('HD5', False)):
            from pretraining_utils import convert_pytorch_to_mxnet
            data_batch = convert_pytorch_to_mxnet(args, data_batch, batch_size_eval)
            if data_batch is None:
                break
        step_num += 1

        data_list = split_and_load(data_batch, ctx)
        ns_label_list, ns_pred_list = [], []
        mask_label_list, mask_pred_list, mask_weight_list = [], [], []
        for data in data_list:
            (input_id, masked_id, masked_position, masked_weight, \
             next_sentence_label, segment_id, valid_length) = data
            valid_length = valid_length.astype(dtype, copy=False)
            out = model(input_id, masked_id, masked_position, masked_weight, \
                        next_sentence_label, segment_id, valid_length)
            classified, decoded, ls1, ls2, num_masks = out
            masked_id = masked_id.reshape(-1)
            ns_label_list.append(next_sentence_label)
            ns_pred_list.append(classified)
            mask_label_list.append(masked_id)
            mask_pred_list.append(decoded)
            mask_weight_list.append(masked_weight)

            valid_length = valid_length.astype('float32', copy=False)
            running_mlm_loss += ls1.as_in_context(mx.cpu())
            running_nsp_loss += ls2.as_in_context(mx.cpu())
            running_num_tks += valid_length.sum().as_in_context(mx.cpu())
        nsp_metric.update(ns_label_list, ns_pred_list)
        mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)

        # logging
        if (step_num + 1) % (log_interval) == 0:
            total_mlm_loss += running_mlm_loss
            total_nsp_loss += running_nsp_loss
            log(begin_time, running_num_tks, running_mlm_loss, running_nsp_loss,
                step_num, mlm_metric, nsp_metric, None, log_interval)
            begin_time = time.time()
            running_mlm_loss = running_nsp_loss = running_num_tks = 0
            mlm_metric.reset_local()
            nsp_metric.reset_local()

    mx.nd.waitall()
    eval_end_time = time.time()
    # accumulate losses from last few batches, too
    if running_mlm_loss != 0:
        total_mlm_loss += running_mlm_loss
        total_nsp_loss += running_nsp_loss
    if step_num > 0:
        total_mlm_loss /= step_num
        total_nsp_loss /= step_num
        logging.info('Eval mlm_loss={:.3f}\tmlm_acc={:.1f}\tnsp_loss={:.3f}\tnsp_acc={:.1f}\t'
                     .format(total_mlm_loss.asscalar(), mlm_metric.get_global()[1] * 100,
                             total_nsp_loss.asscalar(), nsp_metric.get_global()[1] * 100))
        logging.info('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))
    return total_mlm_loss


def generate_dev_set(tokenizer, vocab, cache_file, args):
    """Generate validation set."""
    # set random seed to generate dev data deterministically
    np.random.seed(0)
    random.seed(0)
    mx.random.seed(0)
    worker_pool = multiprocessing.Pool()
    eval_files = nlp.utils.glob(args.data_eval)
    num_files = len(eval_files)
    assert num_files > 0, 'Number of eval files must be greater than 0.' \
                          'Only found %d files at %s'%(num_files, args.data_eval)
    logging.info('Generating validation set from %d files on rank 0.', len(eval_files))
    create_training_instances((eval_files, tokenizer, args.max_seq_length,
                               args.short_seq_prob, args.masked_lm_prob,
                               args.max_predictions_per_seq,
                               args.whole_word_mask, vocab,
                               1, args.num_data_workers,
                               worker_pool, cache_file))
    logging.info('Done generating validation set on rank 0.')

def profile(curr_step, start_step, end_step, profile_name='profile.json',
            early_exit=True):
    """profile the program between [start_step, end_step)."""
    if curr_step == start_step:
        mx.nd.waitall()
        mx.profiler.set_config(profile_memory=False, profile_symbolic=True,
                               profile_imperative=True, filename=profile_name,
                               aggregate_stats=True)
        mx.profiler.set_state('run')
    elif curr_step == end_step:
        mx.nd.waitall()
        mx.profiler.set_state('stop')
        logging.info(mx.profiler.dumps())
        mx.profiler.dump()
        if early_exit:
            exit()

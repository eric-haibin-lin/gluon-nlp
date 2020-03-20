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
"""BERT models."""
# pylint: disable=too-many-lines

import os

import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo import model_store

###############################################################################
#                              COMPONENTS                                     #
###############################################################################


class DotProductSelfAttentionCell(HybridBlock):
    r"""Multi-head Dot Product Self Attention Cell.
    In the DotProductSelfAttentionCell, the input query/key/value will be linearly projected
    for `num_heads` times with different projection matrices. Each projected key, value, query
    will be used to calculate the attention weights and values. The output of each head will be
    concatenated to form the final output.
    This is a more efficient implementation of MultiHeadAttentionCell with
    DotProductAttentionCell as the base_cell:
    score = <W_q h_q, W_k h_k> / sqrt(dim_q)
    Parameters
    ----------
    units : int
        Total number of projected units for query. Must be divided exactly by num_heads.
    num_heads : int
        Number of parallel attention heads
    use_bias : bool, default True
        Whether to use bias when projecting the query/key/values
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.
    Inputs:
      - **qkv** : Symbol or NDArray
        Query / Key / Value vector. Shape (query_length, batch_size, C_in)
      - **valid_len** : Symbol or NDArray or None, default None
        Valid length of the query/key/value slots. Shape (batch_size, query_length)
    Outputs:
      - **context_vec** : Symbol or NDArray
        Shape (query_length, batch_size, context_vec_dim)
      - **att_weights** : Symbol or NDArray
        Attention weights of multiple heads.
        Shape (batch_size, num_heads, query_length, memory_length)
    """
    def __init__(self, units, num_heads, dropout=0.0, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._num_heads = num_heads
        self._use_bias = use_bias
        self._dropout = dropout
        self.units = units
        with self.name_scope():
            if self._use_bias:
                self.query_bias = self.params.get('query_bias', shape=(self.units,),
                                                 init=bias_initializer)
                self.key_bias   = self.params.get('key_bias', shape=(self.units,),
                                                 init=bias_initializer)
                self.value_bias = self.params.get('value_bias', shape=(self.units,),
                                                 init=bias_initializer)
            weight_shape = (self.units, self.units)
            self.query_weight = self.params.get('query_weight', shape=weight_shape,
                                                init=weight_initializer,
                                                allow_deferred_init=True)
            self.key_weight   = self.params.get('key_weight', shape=weight_shape,
                                                init=weight_initializer,
                                                allow_deferred_init=True)
            self.value_weight = self.params.get('value_weight', shape=weight_shape,
                                                init=weight_initializer,
                                                allow_deferred_init=True)
            self.dropout_layer = nn.Dropout(self._dropout)

    def _collect_params_with_prefix(self, prefix=''):
        # the registered parameter names in v0.8 are the following:
        # prefix_proj_query.weight, prefix_proj_query.bias
        # prefix_proj_value.weight, prefix_proj_value.bias
        # prefix_proj_key.weight, prefix_proj_key.bias
        # this is a temporary fix to keep backward compatibility, due to an issue in MXNet:
        # https://github.com/apache/incubator-mxnet/issues/17220
        if prefix:
            prefix += '.'
        ret = {prefix + 'proj_' + k.replace('_', '.') : v for k, v in self._reg_params.items()}
        for name, child in self._children.items():
            ret.update(child._collect_params_with_prefix(prefix + name))
        return ret

    def hybrid_forward(self, F, qkv, valid_len, query_bias, key_bias, value_bias,
                       query_weight, key_weight, value_weight):
        in_bias = F.concat(query_bias, key_bias, value_bias, dim=0)
        in_weight = F.concat(query_weight, key_weight, value_weight, dim=0)
        # qkv_proj shape = (seq_length, batch_size, num_heads * head_dim * 3)
        qkv_proj = F.FullyConnected(data=qkv, weight=in_weight, bias=in_bias,
                                    num_hidden=self.units*3, no_bias=False, flatten=False)
        att_score = F.contrib.interleaved_matmul_selfatt_qk(qkv_proj, heads=self._num_heads)
        if valid_len is not None:
            valid_len = F.broadcast_axis(F.expand_dims(valid_len, axis=1),
                                         axis=1, size=self._num_heads)
            valid_len = valid_len.reshape(shape=(-1, 0), reverse=True)
            att_weights = F.softmax(att_score, length=valid_len, use_length=True, axis=-1)
        else:
            att_weights = F.softmax(att_score, axis=-1)
        # att_weights shape = (batch_size, seq_length, seq_length)
        att_weights = self.dropout_layer(att_weights)
        context_vec = F.contrib.interleaved_matmul_selfatt_valatt(qkv_proj, att_weights,
                                                                  heads=self._num_heads)
        att_weights = att_weights.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True)
        return context_vec, att_weights

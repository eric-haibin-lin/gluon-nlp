# coding: utf-8

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
"""Encoder and decoder usded in sequence-to-sequence learning."""
__all__ = ['GNMTEncoder', 'GNMTDecoder', 'get_gnmt_encoder_decoder']

import mxnet as mx
from mxnet.base import _as_list
from mxnet.gluon import nn, rnn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.rnn.rnn_cell import HybridRecurrentCell, _cells_begin_state, \
                                     _format_sequence, _get_begin_state, _cells_state_info
try:
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell, \
        _get_cell_type, _nested_sequence_last
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell, \
        _get_cell_type, _nested_sequence_last

class HybridBidirectionalCell(HybridRecurrentCell):
    """Bidirectional RNN cell.

    Parameters
    ----------
    l_cell : RecurrentCell
        Cell for forward unrolling
    r_cell : RecurrentCell
        Cell for backward unrolling
    """
    def __init__(self, l_cell, r_cell, output_prefix='bi_'):
        super(HybridBidirectionalCell, self).__init__(prefix='', params=None)
        self.register_child(l_cell, 'l_cell')
        self.register_child(r_cell, 'r_cell')
        self._output_prefix = output_prefix

    def __call__(self, inputs, states):
        raise NotImplementedError("HybridBidirectional cannot be stepped. Please use unroll")

    def __repr__(self):
        s = '{name}(forward={l_cell}, backward={r_cell})'
        return s.format(name=self.__class__.__name__,
                        l_cell=self._children['l_cell'],
                        r_cell=self._children['r_cell'])

    def state_info(self, batch_size=0):
        return _cells_state_info(self._children.values(), batch_size)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._children.values(), **kwargs)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
               valid_length=None):
        # pylint: disable=too-many-locals
        raise NotImplementedError("Please use hybrid_unroll instead")

    # input
    def hybrid_unroll(self, F, inputs, begin_state=None, layout='NTC', merge_outputs=None,
                      valid_length=None):
        # pylint: disable=too-many-locals
        self.reset()
        assert merge_outputs, "Only merge_outputs = True is supported"
        assert valid_length is None, "Only supports valid_length = None"
        batch_size = 1
        if valid_length is None:
            reversed_inputs = F.SequenceReverse(inputs, axis=0)
        begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)

        states = begin_state
        l_cell, r_cell = self._children.values()
        assert merge_outputs, "Only merge_outputs = True is supported"
        l_outputs, l_states = hybrid_unroll(l_cell, inputs=inputs,
                                            begin_state=states[:len(l_cell.state_info(batch_size))],
                                            layout=layout,
                                            valid_length=valid_length)
        r_outputs, r_states = hybrid_unroll(r_cell,
                                            inputs=reversed_inputs,
                                            begin_state=states[len(l_cell.state_info(batch_size)):],
                                            layout=layout,
                                            valid_length=valid_length)

        if valid_length is None:
            reversed_r_outputs = F.SequenceReverse(F.swapaxes(r_outputs,0,1), axis=0)
        if merge_outputs:
            reversed_r_outputs = F.swapaxes(reversed_r_outputs,0,1)
            outputs = F.concat(l_outputs, reversed_r_outputs, dim=2,
                               name='%sout'%self._output_prefix)

        states = l_states + r_states
        return outputs, states

def hybrid_unroll(cell, inputs, begin_state, drop_inputs=0, drop_outputs=0,
           layout='TNC', valid_length=None):
    """Unrolls an RNN cell across time steps.
     Currently, 'TNC' is a preferred layout. unroll on the input of this layout
    runs much faster.
     Parameters
    ----------
    cell : an object whose base class is RNNCell.
        The RNN cell to run on the input sequence.
    inputs : Symbol
        It should have shape (batch_size, length, ...) if `layout` is 'NTC',
        or (length, batch_size, ...) if `layout` is 'TNC'.
    begin_state : nested list of Symbol
        The initial states of the RNN sequence.
    drop_inputs : float, default 0.
        The dropout rate for inputs. Won't apply dropout if it equals 0.
    drop_outputs : float, default 0.
        The dropout rate for outputs. Won't apply dropout if it equals 0.
    layout : str, optional
        `layout` of input symbol. Only used if inputs
        is a single Symbol.
    valid_length : Symbol, NDArray or None
        `valid_length` specifies the length of the sequences in the batch without padding.
        This option is especially useful for building sequence-to-sequence models where
        the input and output sequences would potentially be padded.
        If `valid_length` is None, all sequences are assumed to have the same length.
        If `valid_length` is a Symbol or NDArray, it should have shape (batch_size,).
        The ith element will be the length of the ith sequence in the batch.
        The last valid state will be return and the padded outputs will be masked with 0.
        Note that `valid_length` must be smaller or equal to `length`.
     Returns
    -------
    outputs : Symbol
        the output of the RNN from this unrolling.
     states : list of Symbol
        The new state of this RNN after this unrolling.
        The type of this symbol is same as the output of `begin_state`.
     Examples
    --------
    >>> seq_len = 3
    >>> batch_size = 2
    >>> input_size = 5
    >>> cell = mx.gluon.rnn.LSTMCell(input_size, prefix='rnn_')
    >>> cell.initialize(ctx=mx.cpu())
    >>> rnn_data = mx.nd.normal(loc=0, scale=1, shape=(seq_len, batch_size, input_size))
    >>> state_shape = (batch_size, input_size)
    >>> states = [mx.nd.normal(loc=0, scale=1, shape=state_shape) for i in range(2)]
    >>> valid_length = mx.nd.array([2, 3])
    >>> output, states = mx.gluon.contrib.rnn.rnn_cell.unroll(cell, rnn_data, states,
                                                              valid_length=valid_length,
                                                              layout='TNC')
    >>> print(output)
    [[[ 0.00767238  0.00023103  0.03973929 -0.00925503 -0.05660512]
      [ 0.00881535  0.05428379 -0.02493718 -0.01834097  0.02189514]]
     [[-0.00676967  0.01447039  0.01287002 -0.00574152 -0.05734247]
      [ 0.01568508  0.02650866 -0.04270559 -0.04328435  0.00904011]]
     [[ 0.          0.          0.          0.          0.        ]
      [ 0.01055336  0.02734251 -0.03153727 -0.03742751 -0.01378113]]]
     <NDArray 3x2x5 @cpu(0)>
    """
     # Merge is always True, so we don't need length.
    inputs, axis, F, _ = _format_sequence(0, inputs, layout, True)
    if axis != 0:
        axes = list(range(len(layout)))
        tmp = axes[0]
        axes[0] = axes[axis]
        axes[axis] = tmp
        inputs = F.transpose(inputs, axes=axes)
    states = begin_state
    if drop_inputs:
        inputs = F.Dropout(inputs, p=drop_inputs, axes=(axis,))
    if valid_length is None:
        def loop_body(inputs, states):
            return cell(inputs, states)
    else:
        zeros = []
        for s in states:
            zeros.append(F.zeros_like(s))
        states = list(_as_list(states))
        states.append(F.zeros((1), ctx=zeros[0].context))
        def loop_body(inputs, states):
            cell_states = states[:-1]
            iter_no = states[-1]
            out, new_states = cell(inputs, cell_states)
            for i, state in enumerate(cell_states):
                new_states[i] = F.where(F.broadcast_greater(valid_length, iter_no),
                                        new_states[i], state)
            new_states.append(iter_no + 1)
            return out, new_states
    outputs, states = F.contrib.foreach(loop_body, inputs, states)
    if drop_outputs:
        outputs = F.Dropout(outputs, p=drop_outputs, axes=(axis,))
    if valid_length is not None:
        if axis != 0:
            outputs = F.transpose(outputs, axes)
        outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                 use_sequence_length=True, axis=axis)
        # the last state is the iteration number. We don't need it.
        return outputs, states[:-1]
    else:
        if axis != 0:
            outputs = F.transpose(outputs, axes)
        return outputs, states


class GNMTEncoder(HybridBlock, Seq2SeqEncoder):
    r"""Structure of the RNN Encoder similar to that used in
     "[Arxiv2016] Google's Neural Machine Translation System:
                 Bridgeing the Gap between Human and Machine Translation"

    The encoder first stacks several bidirectional RNN layers and then stacks multiple
    uni-directional RNN layers with residual connections.

    Parameters
    ----------
    cell_type : str or function
        Can be "lstm", "gru" or constructor functions that can be directly called,
         like rnn.LSTMCell
    num_layers : int
        Total number of layers
    num_bi_layers : int
        Total number of bidirectional layers
    hidden_size : int
        Number of hidden units
    dropout : float
        The dropout rate
    use_residual : bool
        Whether to use residual connection. Residual connection will be added in the
        uni-directional RNN layers
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', num_layers=2, num_bi_layers=1, hidden_size=128,
                 dropout=0.0, use_residual=True,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(GNMTEncoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        assert num_bi_layers <= num_layers,\
            'Number of bidirectional layers must be smaller than the total number of layers, ' \
            'num_bi_layers={}, num_layers={}'.format(num_bi_layers, num_layers)
        self._num_bi_layers = num_bi_layers
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._use_residual = use_residual
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            for i in range(num_layers):
                if i < num_bi_layers:
                    self.rnn_cells.add(HybridBidirectionalCell(
                        l_cell=self._cell_type(hidden_size=self._hidden_size,
                                               i2h_weight_initializer=i2h_weight_initializer,
                                               h2h_weight_initializer=h2h_weight_initializer,
                                               i2h_bias_initializer=i2h_bias_initializer,
                                               h2h_bias_initializer=h2h_bias_initializer,
                                               prefix='rnn%d_l_' % i),
                        r_cell=self._cell_type(hidden_size=self._hidden_size,
                                               i2h_weight_initializer=i2h_weight_initializer,
                                               h2h_weight_initializer=h2h_weight_initializer,
                                               i2h_bias_initializer=i2h_bias_initializer,
                                               h2h_bias_initializer=h2h_bias_initializer,
                                               prefix='rnn%d_r_' % i)))
                else:
                    self.rnn_cells.add(
                        self._cell_type(hidden_size=self._hidden_size,
                                        i2h_weight_initializer=i2h_weight_initializer,
                                        h2h_weight_initializer=h2h_weight_initializer,
                                        i2h_bias_initializer=i2h_bias_initializer,
                                        h2h_bias_initializer=h2h_bias_initializer,
                                        prefix='rnn%d_' % i))

    def __call__(self, inputs, states=None):
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the last RNN layer
            - new_states of all the RNN layers
        """
        return super(GNMTEncoder, self).__call__(inputs, states)

    def hybrid_forward(self, F, inputs, states=None):
        assert states is not None
        new_states = []
        outputs = inputs
        for i, cell in enumerate(self.rnn_cells):
            begin_state = None if states is None else states[i]
            if i < self._num_bi_layers:
                # For bidirectional RNN, we use the states of the backward RNN
                outputs, layer_states = cell.hybrid_unroll(
                    F, inputs=inputs, begin_state=begin_state, merge_outputs=True,
                    layout='NTC')
                new_states.append(layer_states[len(self.rnn_cells[i].state_info()) // 2:])
            else:
                outputs, layer_states = mx.gluon.contrib.rnn.rnn_cell.unroll(cell, inputs, begin_state,
                                                                             layout='NTC')
                new_states.append(layer_states)
            # Apply Dropout
            outputs = self.dropout_layer(outputs)
            if self._use_residual:
                if i > self._num_bi_layers:
                    outputs = outputs + inputs
            inputs = outputs
        return [outputs, new_states], []


class GNMTDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the RNN Encoder similar to that used in the
    Google Neural Machine Translation paper.

    We use gnmt_v2 strategy in tensorflow/nmt

    Parameters
    ----------
    cell_type : str or type
    attention_cell : AttentionCell or str
        Arguments of the attention cell.
        Can be 'scaled_luong', 'normed_mlp', 'dot'
    num_layers : int
    hidden_size : int
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, cell_type='lstm', attention_cell='scaled_luong',
                 num_layers=2, hidden_size=128,
                 dropout=0.0, use_residual=True, output_attention=False,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 prefix=None, params=None):
        super(GNMTDecoder, self).__init__(prefix=prefix, params=params)
        self._cell_type = _get_cell_type(cell_type)
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.attention_cell = _get_attention_cell(attention_cell, units=hidden_size)
            self.dropout_layer = nn.Dropout(dropout)
            self.rnn_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.rnn_cells.add(
                    self._cell_type(hidden_size=self._hidden_size,
                                    i2h_weight_initializer=i2h_weight_initializer,
                                    h2h_weight_initializer=h2h_weight_initializer,
                                    i2h_bias_initializer=i2h_bias_initializer,
                                    h2h_bias_initializer=h2h_bias_initializer,
                                    prefix='rnn%d_' % i))

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        """Initialize the state from the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
            The decoder states, includes:

            - rnn_states : NDArray
            - attention_vec : NDArray
            - mem_value : NDArray
            - mem_masks : NDArray, optional
        """
        mem_value, rnn_states = encoder_outputs
        batch_size, _, mem_size = mem_value.shape
        attention_vec = mx.nd.zeros(shape=(batch_size, mem_size), ctx=mem_value.context)
        decoder_states = [rnn_states, attention_vec, mem_value]
        mem_length = mem_value.shape[1]
        if encoder_valid_length is not None:
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(mem_masks)
        return decoder_states

    def decode_seq(self, inputs, states, valid_length=None):
        """Decode the decoder inputs. This function is only used for training.

        Parameters
        ----------
        inputs : NDArray, Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial decoder states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        output : NDArray, Shape (batch_size, length, C_out)
        states : list
            The decoder states, includes:

            - rnn_states : NDArray
            - attention_vec : NDArray
            - mem_value : NDArray
            - mem_masks : NDArray, optional
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        inputs = inputs.swapaxes(0, 1)
        output = []
        additional_outputs = []
        def body(input_i, states):
            return self.forward(input_i, states)
        output, states = mx.nd.contrib.foreach(body, inputs, states)
        return output.swapaxes(0,1)

    def __call__(self, step_input, states): #pylint: disable=arguments-differ
        """One-step-ahead decoding of the GNMT decoder.

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : list of NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, C_out)
        new_states: list
            Includes

            - rnn_states : list of NDArray or Symbol
            - attention_vec : NDArray or Symbol, Shape (batch_size, C_memory)
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)
        """
        return super(GNMTDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states):  #pylint: disable=arguments-differ, missing-docstring
        #step_output, new_states, step_additional_outputs =\
        #    super(GNMTDecoder, self).forward(step_input, states)
        # In hybrid_forward, only the rnn_states and attention_vec are calculated.
        # We directly append the mem_value and mem_masks in the forward() function.
        # We apply this trick because the memory value/mask can be directly appended to the next
        # timestamp and there is no need to create additional NDArrays. If we use HybridBlock,
        # new NDArrays will be created even for identity mapping.
        # See https://github.com/apache/incubator-mxnet/issues/10167
        step_output, new_states =\
            super(GNMTDecoder, self).forward(step_input, states)
        new_states += states[2:]
        return step_output, new_states#, step_additional_outputs

    def hybrid_forward(self, F, step_input, states):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol
        states : list of NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, C_out)
        new_states: list
            Includes

            - rnn_states : list of NDArray or Symbol
            - attention_vec : NDArray or Symbol, Shape (batch_size, C_memory)

        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, 1, mem_length) or
            (batch_size, num_heads, 1, mem_length)

        """
        has_mem_mask = (len(states) == 4)
        if has_mem_mask:
            rnn_states, attention_output, mem_value, mem_masks = states
            mem_masks = F.expand_dims(mem_masks, axis=1)
        else:
            rnn_states, attention_output, mem_value = states
            mem_masks = None
        new_rnn_states = []
        # Process the first layer
        rnn_out, layer_state =\
            self.rnn_cells[0](F.concat(step_input, attention_output, dim=-1), rnn_states[0])
        new_rnn_states.append(layer_state)
        attention_vec, attention_weights =\
            self.attention_cell(F.expand_dims(rnn_out, axis=1),  # Shape(B, 1, C)
                                mem_value,
                                mem_value,
                                mem_masks)
        attention_vec = F.reshape(attention_vec, shape=(0, -1))
        # Process the 2nd layer - the last layer
        for i in range(1, len(self.rnn_cells)):
            curr_input = rnn_out
            rnn_cell = self.rnn_cells[i]
            # Concatenate the attention vector calculated by the bottom layer and the output of the
            # previous layer
            rnn_out, layer_state = rnn_cell(F.concat(curr_input, attention_vec, dim=-1),
                                            rnn_states[i])
            rnn_out = self.dropout_layer(rnn_out)
            if self._use_residual:
                rnn_out = rnn_out + curr_input
            # Append new RNN state
            new_rnn_states.append(layer_state)
        new_states = [new_rnn_states, attention_vec]
        return rnn_out, new_states


def get_gnmt_encoder_decoder(cell_type='lstm', attention_cell='scaled_luong', num_layers=2,
                             num_bi_layers=1, hidden_size=128, dropout=0.0, use_residual=False,
                             i2h_weight_initializer=None, h2h_weight_initializer=None,
                             i2h_bias_initializer=mx.init.LSTMBias(forget_bias=1.0),
                             h2h_bias_initializer='zeros',
                             prefix='gnmt_', params=None):
    """Build a pair of GNMT encoder/decoder

    Parameters
    ----------
    cell_type : str or type
    attention_cell : str or AttentionCell
    num_layers : int
    num_bi_layers : int
    hidden_size : int
    dropout : float
    use_residual : bool
    i2h_weight_initializer : mx.init.Initializer or None
    h2h_weight_initializer : mx.init.Initializer or None
    i2h_bias_initializer : mx.init.Initializer or None
    h2h_bias_initializer : mx.init.Initializer or None
    prefix : str, default 'gnmt_'
        Prefix for name of `Block`s.
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.

    Returns
    -------
    encoder : GNMTEncoder
    decoder : GNMTDecoder
    """
    encoder = GNMTEncoder(cell_type=cell_type, num_layers=num_layers, num_bi_layers=num_bi_layers,
                          hidden_size=hidden_size, dropout=dropout,
                          use_residual=use_residual,
                          i2h_weight_initializer=i2h_weight_initializer,
                          h2h_weight_initializer=h2h_weight_initializer,
                          i2h_bias_initializer=i2h_bias_initializer,
                          h2h_bias_initializer=h2h_bias_initializer,
                          prefix=prefix + 'enc_', params=params)
    decoder = GNMTDecoder(cell_type=cell_type, attention_cell=attention_cell, num_layers=num_layers,
                          hidden_size=hidden_size, dropout=dropout,
                          use_residual=use_residual,
                          i2h_weight_initializer=i2h_weight_initializer,
                          h2h_weight_initializer=h2h_weight_initializer,
                          i2h_bias_initializer=i2h_bias_initializer,
                          h2h_bias_initializer=h2h_bias_initializer,
                          prefix=prefix + 'dec_', params=params)
    return encoder, decoder

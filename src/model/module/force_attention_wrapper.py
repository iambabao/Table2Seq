# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 21:39
@Desc       :
"""

import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class ForceAttentionWrapperState(collections.namedtuple('ForceAttentionWrapperState',
                                                        ('cell_state', 'time',
                                                         'word_coverage', 'attr_coverage',
                                                         'alignment_history'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(ForceAttentionWrapperState, self)._replace(**kwargs)
        )


class ForceAttentionWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 memory,
                 attr_input_ids,
                 attention_size,
                 attr_size,
                 memory_sequence_length=None,
                 initial_cell_state=None,
                 alignment_history=False,
                 name=None):
        super(ForceAttentionWrapper, self).__init__(name=name)
        self._cell = cell
        self._memory = memory
        self._attr_input_ids = attr_input_ids
        self._attention_size = attention_size
        self._attr_size = attr_size
        self._memory_sequence_length = memory_sequence_length
        self._initial_cell_state = initial_cell_state
        self._memory_state_size = self._memory.shape[-1].value
        self._max_memory_sequence_length = tf.reduce_max(self._memory_sequence_length)
        self._alignment_history = alignment_history

        self._memory_layer = tf.layers.Dense(self._attention_size, use_bias=False, name='memory_layer')
        self._query_layer = tf.layers.Dense(self._attention_size, use_bias=False, name='query_layer')
        self._attention_v = tf.get_variable('attention_v', self._attention_size, dtype=tf.float32)
        self._pi = tf.get_variable('pi', [],
                                   initializer=tf.keras.initializers.random_uniform(0, 1, dtype=tf.float32),
                                   constraint=lambda x: tf.clip_by_value(x, 0, 1),
                                   dtype=tf.float32)

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, ForceAttentionWrapperState):
            raise TypeError(
                'Expected state to be instance of ForceAttentionWrapper. Received type {} instead.'.format(type(state))
            )
        prev_cell_state = state.cell_state
        prev_time = state.time
        prev_word_coverage = state.word_coverage
        prev_attr_coverage = state.attr_coverage
        prev_alignment_history = state.alignment_history

        _, cell_state = self._cell(inputs, prev_cell_state, scope)

        word_attention = self._get_word_attention(self._memory, cell_state.h)  # [batch_size, seq_len]
        attr_attention = self._get_attr_attention(word_attention)  # [batch_size, attr_size]

        word_coverage = prev_word_coverage + word_attention
        word_coverage_mean = word_coverage / tf.cast(prev_time + 1, dtype=tf.float32)
        attr_coverage = prev_attr_coverage + attr_attention
        attr_coverage_mean = attr_coverage / tf.cast(prev_time + 1, dtype=tf.float32)

        # expand attr_coverage_mean to [batch_size, seq_len] for calculating zeta
        expanded_attr_coverage_mean = tf.gather(attr_coverage_mean, self._attr_input_ids, batch_dims=1)
        zeta = tf.minimum(word_coverage_mean, expanded_attr_coverage_mean)
        zeta = tf.reduce_max(zeta, axis=-1, keepdims=True) - zeta  # [batch_size, seq_len]

        c = tf.reduce_sum(self._memory * tf.expand_dims(word_attention, axis=-1), axis=1)
        v = tf.reduce_sum(self._memory * tf.expand_dims(zeta, axis=-1), axis=1)
        c_tilde = self._pi * c + (1 - self._pi) * v  # [batch_size, hidden_size]
        outputs = tf.concat([cell_state.h, c_tilde], axis=-1)

        if self._alignment_history:
            alignment_history = prev_alignment_history.write(prev_time, word_attention)
        else:
            alignment_history = prev_alignment_history
        state = ForceAttentionWrapperState(cell_state=cell_state, time=prev_time + 1,
                                           word_coverage=word_coverage, attr_coverage=attr_coverage,
                                           alignment_history=alignment_history)
        return outputs, state

    def _get_word_attention(self, memory, query):
        memory = self._memory_layer(memory)
        query = tf.expand_dims(self._query_layer(query), axis=1)
        word_attention = tf.reduce_sum(self._attention_v * tf.math.tanh(memory + query), axis=-1)

        if self._memory_sequence_length is not None:
            mask = tf.sequence_mask(self._memory_sequence_length, maxlen=self._max_memory_sequence_length)
            mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)
            word_attention += -1e9 * mask

        word_attention = tf.math.softmax(word_attention, axis=-1)  # [batch_size, seq_len]

        return word_attention

    def _get_attr_attention(self, word_attention):
        attr_input_em = tf.one_hot(self._attr_input_ids, self._attr_size)
        expanded_word_attention = tf.expand_dims(word_attention, axis=-1)
        expanded_word_attention = attr_input_em * expanded_word_attention  # [batch_size, seq_len, attr_size]

        attr_num = tf.reduce_sum(attr_input_em, axis=1)
        attr_attention = tf.reduce_sum(expanded_word_attention, axis=1)
        attr_attention = tf.div_no_nan(attr_attention, attr_num)  # [batch_size, attr_size]

        return attr_attention

    @property
    def state_size(self):
        """
            size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return ForceAttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            word_coverage=self._max_memory_sequence_length,
            attr_coverage=self._attr_size,
            alignment_history=()
        )

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._cell.output_size + self._memory_state_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            time = tf.zeros([], tf.int32)
            word_coverage = tf.zeros([batch_size, self._max_memory_sequence_length], tf.float32)
            attr_coverage = tf.zeros([batch_size, self._attr_size], tf.float32)
            alignment_history = tf.TensorArray(tf.float32, size=0, dynamic_size=True) if self._alignment_history else ()
            return ForceAttentionWrapperState(cell_state=cell_state, time=time,
                                              word_coverage=word_coverage, attr_coverage=attr_coverage,
                                              alignment_history=alignment_history)

import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util


class FAWrapperState(collections.namedtuple('FAWrapperState',
                                            ('cell_state', 'time',
                                             'word_coverage', 'attr_coverage'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(FAWrapperState, self)._replace(**kwargs)
        )


class FAWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 encoder_outputs,
                 attr_size,
                 attr_input_ids,
                 memory_sequence_length=None,
                 encoder_state_size=None,
                 initial_cell_state=None,
                 name=None):
        super(FAWrapper, self).__init__(name=name)
        self._cell = cell

        self._encoder_outputs = encoder_outputs
        self._attr_size = attr_size
        self._attr_input_ids = attr_input_ids
        self._seq_len = tf.shape(attr_input_ids)[-1]
        self._memory_sequence_length = memory_sequence_length
        if encoder_state_size is None:
            encoder_state_size = self._encoder_outputs.shape[-1].value
            if encoder_state_size is None:
                raise ValueError('encoder_state_size must be set.')
        self._initial_cell_state = initial_cell_state

        self._weights_p = tf.get_variable('weights_p', [encoder_state_size, 1])
        self._weights_q = tf.get_variable('weights_q', [encoder_state_size, 1])
        self._pi = tf.get_variable(
            initializer=tf.keras.initializers.random_uniform(0, 1, dtype=tf.float32),
            shape=[],
            name='pi'
        )

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, FAWrapperState):
            raise TypeError(
                'Expected state to be instance of FAWrapper. Received type {} instead.'.format(type(state))
            )
        prev_cell_state = state.cell_state
        time = state.time + 1
        prev_word_coverage = state.word_coverage
        prev_attr_coverage = state.attr_coverage

        word_attention = self._get_word_attention(self._encoder_outputs, prev_cell_state.h)  # [batch_size, seq_len]
        attr_attention = self._get_attr_attention(word_attention)  # [batch_size, attr_size]

        word_coverage = prev_word_coverage + word_attention
        word_coverage_mean = word_coverage / tf.cast(time, dtype=tf.float32)
        attr_coverage = prev_attr_coverage + attr_attention
        attr_coverage_mean = attr_coverage / tf.cast(time, dtype=tf.float32)

        # expand attr_coverage_mean to [batch_size, seq_len] for calculating zeta
        expanded_attr_coverage_mean = tf.gather(attr_coverage_mean, self._attr_input_ids, batch_dims=1)
        zeta = tf.minimum(word_coverage_mean, expanded_attr_coverage_mean)
        zeta = tf.reduce_max(zeta, axis=-1, keepdims=True) - zeta  # [batch_size, seq_len]

        # v = tf.einsum('ijk,ij->ik', self._encoder_outputs, zeta)  # version 1
        v = tf.reduce_sum(self._encoder_outputs * tf.expand_dims(zeta, axis=-1), axis=1)  # version 2: memory efficiency
        c = self._pi * prev_cell_state.c + (1 - self._pi) * v  # [batch_size, hidden_size]

        outputs, cell_state = self._cell(inputs, tf.nn.rnn_cell.LSTMStateTuple(c=c, h=prev_cell_state.h), scope)
        state = FAWrapperState(cell_state=cell_state, time=time,
                               word_coverage=word_coverage, attr_coverage=attr_coverage)
        return outputs, state

    def _get_word_attention(self, memory, query):
        w1 = tf.reshape(tf.matmul(memory, self._weights_p), [-1, self._seq_len])
        w2 = tf.reshape(tf.matmul(query, self._weights_q), [-1, 1])
        word_attention = tf.math.tanh(w1 + w2)  # [batch_size, seq_len]

        if self._memory_sequence_length is not None:
            mask = tf.sequence_mask(self._memory_sequence_length, maxlen=self._seq_len)
            mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)
            word_attention += -1e9 * mask

        word_attention = tf.math.softmax(word_attention, axis=-1)

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
        return FAWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            word_coverage=self._seq_len,
            attr_coverage=self._attr_size
        )

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            time = tf.zeros([], tf.int32)
            word_coverage = tf.zeros([batch_size, self._seq_len], tf.float32)
            attr_coverage = tf.zeros([batch_size, self._attr_size], tf.float32)
            return FAWrapperState(cell_state=cell_state, time=time,
                                  word_coverage=word_coverage, attr_coverage=attr_coverage)

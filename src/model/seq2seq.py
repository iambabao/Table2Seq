# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 21:39
@Desc       :
"""

import tensorflow as tf

from .module.metrics import get_loss, get_accuracy


class Seq2Seq:
    def __init__(self, config, word_embedding_matrix):
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.vocab_size = config.vocab_size
        self.oov_vocab_size = config.oov_vocab_size
        self.attr_size = config.attr_size
        self.pos_size = config.pos_size
        self.max_seq_len = config.sequence_len

        self.word_em_size = config.word_em_size
        self.attr_em_size = config.attr_em_size
        self.pos_em_size = config.pos_em_size
        self.hidden_size = config.hidden_size

        self.lr = config.lr
        self.dropout = config.dropout

        self.value_inp = tf.placeholder(tf.int32, [None, None], name='value_inp')
        self.attr_inp = tf.placeholder(tf.int32, [None, None], name='attr_inp')
        self.pos_fw_inp = tf.placeholder(tf.int32, [None, None], name='pos_fw_inp')
        self.pos_bw_inp = tf.placeholder(tf.int32, [None, None], name='pos_bw_inp')
        self.desc_inp = tf.placeholder(tf.int32, [None, None], name='desc_inp')
        self.desc_out = tf.placeholder(tf.int32, [None, None], name='desc_out')
        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')
        self.training = tf.placeholder(tf.bool, [], name='training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if word_embedding_matrix is not None:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size + self.oov_vocab_size,
                self.word_em_size,
                embeddings_initializer=tf.constant_initializer(word_embedding_matrix),
                name='word_embedding'
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size + self.oov_vocab_size,
                self.word_em_size,
                name='word_embedding'
            )
        self.attr_embedding = tf.keras.layers.Embedding(self.attr_size, self.attr_em_size, name='attr_embedding')
        self.pos_embedding = tf.keras.layers.Embedding(self.pos_size, self.pos_em_size, name='pos_embedding')
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.final_dense = tf.layers.Dense(self.vocab_size, name='final_dense')
        self.transparent_layer = tf.keras.layers.Dropout(0)

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            assert False

        logits, self.predicted_ids, self.alignment_history = self.forward()

        clipped_desc_out = tf.where(
            tf.greater_equal(self.desc_out, self.vocab_size),
            tf.ones_like(self.desc_out) * config.unk_id,
            self.desc_out
        )
        self.loss = get_loss(clipped_desc_out, logits)
        self.accu = get_accuracy(clipped_desc_out, logits)
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # embedding
        src_em = self.src_embedding_layer()
        tgt_em = self.tgt_embedding_layer()

        # encoding
        enc_output, enc_state = self.encoding_layer(src_em)

        # decoding in training
        logits = self.training_decoding_layer(enc_output, enc_state, tgt_em)

        # decoding in testing
        predicted_ids, alignment_history = self.inference_decoding_layer(enc_output, enc_state, self.src_len)

        return logits, predicted_ids, alignment_history

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def src_embedding_layer(self):
        with tf.device('/cpu:0'):
            value_em = self.word_embedding(self.value_inp)
            attr_em = self.attr_embedding(self.attr_inp)
            pos_fw_em = self.pos_embedding(self.pos_fw_inp)
            pos_bw_em = self.pos_embedding(self.pos_bw_inp)
        src_em = tf.concat([value_em, attr_em, pos_fw_em, pos_bw_em], axis=-1)
        src_em = self.embedding_dropout(src_em, training=self.training)

        return src_em

    def tgt_embedding_layer(self):
        with tf.device('/cpu:0'):
            tgt_em = self.word_embedding(self.desc_inp)

        return tgt_em

    def encoding_layer(self, src_em):
        with tf.variable_scope('encoder'):
            enc_output, enc_state = tf.nn.dynamic_rnn(
                self.encoder_cell,
                src_em,
                self.src_len,
                dtype=tf.float32
            )

        return enc_output, enc_state

    def training_decoding_layer(self, enc_output, enc_state, tgt_em):
        with tf.variable_scope('decoder', reuse=False):
            # add attention mechanism to decoder cell
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.hidden_size,
                enc_output,
                memory_sequence_length=self.src_len
            )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism,
                attention_layer=self.transparent_layer
            )

            dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
            dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

            # build teacher forcing decoder
            helper = tf.contrib.seq2seq.TrainingHelper(tgt_em, self.tgt_len)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)

            # decoding
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

            logits = final_outputs.rnn_output

        return logits

    def inference_decoding_layer(self, enc_output, enc_state, src_len):
        with tf.variable_scope('decoder', reuse=True):
            # add attention mechanism to decoder cell
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.hidden_size,
                enc_output,
                memory_sequence_length=src_len
            )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism,
                attention_layer=self.transparent_layer,
                alignment_history=True
            )

            dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
            dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

            # build greedy decoder
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.word_embedding,
                tf.fill([tf.shape(self.src_len)[0]], self.sos_id),
                self.eos_id
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)

            # decoding
            final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_seq_len)

            predicted_ids = final_outputs.sample_id
            alignment_history = tf.transpose(final_state.alignment_history.stack(), perm=[1, 0, 2])

        return predicted_ids, alignment_history

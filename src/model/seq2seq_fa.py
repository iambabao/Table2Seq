import tensorflow as tf

from .module.force_attention_wrapper import ForceAttentionWrapper
from .module.metrics import get_loss, get_accuracy


class Seq2SeqFA:
    def __init__(self, config, word_embedding_matrix):
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.vocab_size = config.vocab_size
        self.attr_size = config.attr_size
        self.pos_size = config.pos_size
        self.max_seq_len = config.sequence_len
        self.beam_size = config.top_k
        self.beam_search = config.beam_search

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
                self.vocab_size,
                self.word_em_size,
                embeddings_initializer=tf.constant_initializer(word_embedding_matrix),
                trainable=config.embedding_trainable,
                name='word_embedding'
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(self.vocab_size, self.word_em_size, name='word_embedding')
        self.attr_embedding = tf.keras.layers.Embedding(self.attr_size, self.attr_em_size, name='attr_embedding')
        self.pos_embedding = tf.keras.layers.Embedding(self.pos_size, self.pos_em_size, name='pos_embedding')
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.final_dense = tf.layers.Dense(self.vocab_size, name='final_dense')

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

        logits, coverage_loss, self.predicted_ids = self.forward()
        self.loss = get_loss(self.desc_out, logits)
        # self.loss += coverage_loss  # add coverage loss after 4 epochs according to the paper
        self.accu = get_accuracy(self.desc_out, logits)
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('coverage_loss', coverage_loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # embedding
        src_em = self.src_embedding_layer(training=self.training)
        tgt_em = self.tgt_embedding_layer(training=self.training)

        # encoding
        enc_output, enc_state = self.encoding_layer(src_em)

        # decoding in training
        logits, coverage_loss = self.training_decoding_layer(enc_output, enc_state, tgt_em)

        # decoding in testing
        if not self.beam_search:
            predicted_ids = self.inference_decoding_layer(enc_output, enc_state, beam_search=self.beam_search)
        else:
            # tiled to beam size
            tiled_enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=self.beam_size)
            tiled_enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=self.beam_size)
            predicted_ids = self.inference_decoding_layer(tiled_enc_output, tiled_enc_state, beam_search=self.beam_search)

        return logits, coverage_loss, predicted_ids

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        # gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def src_embedding_layer(self, training):
        with tf.device('/cpu:0'):
            value_em = self.word_embedding(self.value_inp)
            attr_em = self.attr_embedding(self.attr_inp)
            pos_fw_em = self.pos_embedding(self.pos_fw_inp)
            pos_bw_em = self.pos_embedding(self.pos_bw_inp)
        src_em = tf.concat([value_em, attr_em, pos_fw_em, pos_bw_em], axis=-1)
        src_em = self.embedding_dropout(src_em, training=training)

        return src_em

    def tgt_embedding_layer(self, training):
        with tf.device('/cpu:0'):
            desc_em = self.word_embedding(self.desc_inp)
        tgt_em = self.embedding_dropout(desc_em, training=training)

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
        # add force attention mechanism to decoder cell
        with tf.variable_scope('force_attention_mechanism', reuse=False):
            decoder_cell = ForceAttentionWrapper(
                self.decoder_cell,
                enc_output,
                self.attr_inp,
                attention_size=self.hidden_size,
                attr_size=self.attr_size,
                memory_sequence_length=self.src_len
            )

        dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
        dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

        # build teacher forcing decoder
        helper = tf.contrib.seq2seq.TrainingHelper(tgt_em, self.tgt_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)

        with tf.variable_scope('decoder', reuse=False):
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

        logits = final_outputs.rnn_output

        attr_coverage = final_state.attr_coverage
        attr_coverage_mean = attr_coverage / tf.expand_dims(tf.cast(final_sequence_lengths, dtype=tf.float32), axis=-1)
        k = tf.reduce_sum(tf.cast(tf.greater(attr_coverage, 0.0), dtype=tf.float32), axis=-1, keepdims=True)
        coverage_loss = tf.reduce_mean(0.3 * tf.reduce_sum((attr_coverage_mean - tf.div_no_nan(1.0, k))**2, axis=-1))

        return logits, coverage_loss

    def inference_decoding_layer(self, enc_output, enc_state, beam_search):
        # add force attention mechanism to decoder cell
        with tf.variable_scope('force_attention_mechanism', reuse=True):
            decoder_cell = ForceAttentionWrapper(
                self.decoder_cell,
                enc_output,
                self.attr_inp,
                attention_size=self.hidden_size,
                attr_size=self.attr_size,
                memory_sequence_length=self.src_len
            )

        dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
        dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

        if not beam_search:
            # build greedy decoder
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.word_embedding,
                tf.fill([tf.shape(self.src_len)[0]], self.sos_id),
                self.eos_id
            )
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)
        else:
            # build beam search decoder
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.word_embedding,
                start_tokens=tf.fill([tf.shape(self.src_len)[0]], self.sos_id),
                end_token=self.eos_id,
                output_layer=self.final_dense,
                initial_state=dec_initial_state,
                beam_width=self.beam_size,
                length_penalty_weight=0.0,
                coverage_penalty_weight=0.0
            )

        # decoding
        with tf.variable_scope('decoder', reuse=True):
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_seq_len)

        if not beam_search:
            predicted_ids = final_outputs.sample_id
        else:
            predicted_ids = final_outputs.predicted_ids  # (batch_size, seq_len, beam_size)
            predicted_ids = tf.transpose(predicted_ids, perm=[0, 2, 1])  # (batch_size, beam_size, seq_len)
            predicted_ids = predicted_ids[:, 0, :]  # keep top one

        return predicted_ids

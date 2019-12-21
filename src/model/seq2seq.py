import tensorflow as tf

from .module.metrics import get_loss, get_accuracy


class Seq2Seq:
    def __init__(self, config, word_embedding_matrix):
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.word_size = config.word_size
        self.attr_size = config.attr_size
        self.max_seq_len = config.sequence_len
        self.beam_size = config.top_k

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

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if word_embedding_matrix is not None:
            self.word_embedding = tf.keras.layers.Embedding(
                self.word_size,
                self.word_em_size,
                embeddings_initializer=tf.constant_initializer(word_embedding_matrix),
                trainable=config.embedding_trainable,
                name='word_embedding'
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(self.word_size, self.word_em_size, name='word_embedding')
        self.attr_embedding = tf.keras.layers.Embedding(self.attr_size, self.attr_em_size, name='attr_embedding')
        self.pos_embedding = tf.keras.layers.Embedding(self.max_seq_len, self.pos_em_size, name='pos_embedding')
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        self.final_dense = tf.layers.Dense(self.word_size, name='final_dense')

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

        self.gradients, self.train_op, self.loss, self.accu = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.train_summary = tf.summary.merge_all()

        self.greedy_pred_id = self.greedy_inference()
        self.beam_search_pred_id = self.beam_search_inference()

    def get_train_op(self):
        # embedding
        src_em = self.src_embedding_layer(training=True)
        tgt_em = self.tgt_embedding_layer(training=True)

        # encoding
        enc_output, enc_state = self.encoding_layer(src_em, reuse=False)

        # decoding
        logits = self.training_decoding_layer(enc_output, enc_state, tgt_em)

        loss = get_loss(self.desc_out, logits)
        accu = get_accuracy(self.desc_out, logits)

        gradients = tf.gradients(loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op, loss, accu

    # greedy decoding
    def greedy_inference(self):
        # embedding
        src_em = self.src_embedding_layer(training=False)

        # encoding
        enc_output, enc_state = self.encoding_layer(src_em, reuse=True)

        # decoding
        pred_id = self.inference_decoding_layer(enc_output, enc_state, self.src_len, beam_search=False)

        return pred_id

    # beam search decoding
    def beam_search_inference(self):
        # embedding
        src_em = self.src_embedding_layer(training=False)

        # encoding
        enc_output, enc_state = self.encoding_layer(src_em, reuse=True)

        # tiled to beam size
        tiled_enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=self.beam_size)
        tiled_enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=self.beam_size)
        tiled_src_len = tf.contrib.seq2seq.tile_batch(self.src_len, multiplier=self.beam_size)

        # decoding
        pred_id = self.inference_decoding_layer(tiled_enc_output, tiled_enc_state, tiled_src_len, beam_search=False)

        return pred_id

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

    def encoding_layer(self, src_em, reuse):
        with tf.variable_scope('encoder', reuse=reuse):
            enc_output, enc_state = tf.nn.dynamic_rnn(
                self.encoder_cell,
                src_em,
                self.src_len,
                dtype=tf.float32
            )

        return enc_output, enc_state

    def training_decoding_layer(self, enc_output, enc_state, tgt_em):
        # add attention mechanism to decoder cell
        with tf.variable_scope('attention_mechanism', reuse=False):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.hidden_size,
                enc_output,
                memory_sequence_length=self.src_len
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism,
                attention_layer_size=self.hidden_size
            )

        dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(enc_output)[0], dtype=tf.float32)
        dec_initial_state = dec_initial_state.clone(cell_state=enc_state)

        # build teacher forcing decoder
        helper = tf.contrib.seq2seq.TrainingHelper(tgt_em, self.tgt_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state, self.final_dense)

        with tf.variable_scope('decoder', reuse=False):
            dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

        logits = dec_output.rnn_output

        return logits

    def inference_decoding_layer(self, enc_output, enc_state, src_len, beam_search):
        # add attention mechanism to decoder cell
        with tf.variable_scope('attention_mechanism', reuse=True):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.hidden_size,
                enc_output,
                memory_sequence_length=src_len
            )

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,
                attention_mechanism,
                attention_layer_size=self.hidden_size
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
                initial_state=dec_initial_state,
                beam_width=self.beam_size,
                output_layer=self.final_dense,
                length_penalty_weight=0.0,
                coverage_penalty_weight=0.0
            )

        # decoding
        with tf.variable_scope('decoder', reuse=True):
            dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=int(self.max_seq_len / 2))

        if not beam_search:
            pred_id = dec_output.sample_id
        else:
            pred_id = dec_output.predicted_ids  # (batch_size, seq_len, beam_size)
            pred_id = tf.transpose(pred_id, perm=[0, 2, 1])  # (batch_size, beam_size, seq_len)

        return pred_id

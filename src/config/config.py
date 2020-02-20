# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 21:39
@Desc       :
"""

import os


class Config:
    def __init__(self, root_dir, current_model, num_epoch=30, batch_size=32,
                 sequence_len=150, hidden_size=600, top_k=5,
                 word_em_size=500, attr_em_size=50, pos_em_size=10,
                 fc_size_s=128, fc_size_m=512, fc_size_l=1024,
                 optimizer='Adam', lr=1e-3, dropout=0.1):
        self.root_dir = root_dir

        self.temp_dir = os.path.join(self.root_dir, 'temp')

        self.data_dir = os.path.join(self.root_dir, 'data')
        self.train_data = os.path.join(self.data_dir, 'train_1480.json')
        self.valid_data = os.path.join(self.data_dir, 'valid_1480.json')
        self.test_data = os.path.join(self.data_dir, 'test_1480.json')
        # self.test_data = os.path.join(self.data_dir, 'valid_1480_small.json')
        self.valid_data_small = os.path.join(self.data_dir, 'valid_1480_small.json')
        self.word_dict = os.path.join(self.data_dir, 'dict_word.json')
        self.attr_dict = os.path.join(self.data_dir, 'dict_attr.json')

        self.embedding_dir = os.path.join(self.data_dir, 'embedding')
        self.glove_file = os.path.join(self.embedding_dir, 'glove.6B.300d.txt')

        self.current_model = current_model
        self.result_dir = os.path.join(self.root_dir, 'result', self.current_model)
        self.model_file = os.path.join(self.result_dir, 'model')
        self.train_log_dir = os.path.join(self.result_dir, 'train_log')
        self.valid_log_dir = os.path.join(self.result_dir, 'valid_log')
        self.valid_result = os.path.join(self.result_dir, 'valid_result.json')
        self.test_result = os.path.join(self.result_dir, 'test_result.json')

        self.pad = '<pad>'
        self.pad_id = 0
        self.unk = '<unk>'
        self.unk_id = 1
        self.sos = '<sos>'
        self.sos_id = 2
        self.eos = '<eos>'
        self.eos_id = 3
        self.num = '<num>'
        self.num_id = 4
        self.time = '<time>'
        self.time_id = 5
        self.vocab_size = 20000
        self.pos_size = 10
        self.to_lower = True

        self.top_k = top_k
        self.word_em_size = word_em_size
        self.attr_em_size = attr_em_size
        self.pos_em_size = pos_em_size
        self.sequence_len = sequence_len

        # RNN
        self.hidden_size = hidden_size

        # FC
        self.fc_size_s = fc_size_s
        self.fc_size_m = fc_size_m
        self.fc_size_l = fc_size_l

        # Train
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.optimizer = optimizer

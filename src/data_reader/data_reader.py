# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 21:39
@Desc       :
"""

import numpy as np

from src.wikientity import WikiEntity
from src.utils import convert_list


class DataReader:
    def __init__(self, config):
        self.config = config

    def _read_data(self, data_file, max_data_size=None):
        value_seq = []
        attr_seq = []
        pos_fw_seq = []
        pos_bw_seq = []
        desc_seq = []

        counter = 0
        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                we = WikiEntity(line)

                value = []
                attr = []
                pos_fw = []
                pos_bw = []
                box = we.get_box()
                for a in box.keys():
                    v = box[a].split()
                    a = [a] * len(v)
                    p = list(range(len(v)))
                    value += v
                    attr += a
                    pos_fw += p
                    pos_bw += reversed(p)
                desc = we.get_desc().split()

                # check length and limit the maximum length of input
                assert len(value) == len(attr)
                assert len(value) == len(pos_fw)
                assert len(value) == len(pos_bw)
                if len(value) == 0:
                    continue

                value = value[:self.config.sequence_len]
                attr = attr[:self.config.sequence_len]
                pos_fw = pos_fw[:self.config.sequence_len]
                pos_fw = np.minimum(pos_fw, self.config.pos_size - 1).tolist()  # 1 for zero
                pos_bw = pos_bw[:self.config.sequence_len]
                pos_bw = np.minimum(pos_bw, self.config.pos_size - 1).tolist()  # 1 for zero
                desc = desc[:self.config.sequence_len - 2]  # 2 for sos and eos

                if self.config.to_lower:
                    value = list(map(str.lower, value))
                    attr = list(map(str.lower, attr))
                    desc = list(map(str.lower, desc))
                desc = [self.config.sos] + desc + [self.config.eos]

                value_seq.append(convert_list(value, self.config.word_2_id, self.config.pad_id, self.config.unk_id))
                attr_seq.append(convert_list(attr, self.config.attr_2_id, self.config.pad_id, self.config.unk_id))
                pos_fw_seq.append(pos_fw)
                pos_bw_seq.append(pos_bw)
                desc_seq.append(convert_list(desc, self.config.word_2_id, self.config.pad_id, self.config.unk_id))

                counter += 1
                if counter % 10000 == 0:
                    print('\rprocessing file {}: {:>6d}'.format(data_file, counter), end='')
                if max_data_size and counter >= max_data_size:
                    break
            print()

        return value_seq, attr_seq, pos_fw_seq, pos_bw_seq, desc_seq

    def read_train_data(self, max_data_size=None):
        return self._read_data(self.config.train_data, max_data_size)

    def read_valid_data(self, max_data_size=None):
        return self._read_data(self.config.valid_data, max_data_size)

    def read_test_data(self, max_data_size=None):
        return self._read_data(self.config.test_data, max_data_size)

    def read_valid_data_small(self, max_data_size=None):
        return self._read_data(self.config.valid_data_small, max_data_size)

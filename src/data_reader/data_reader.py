import numpy as np

from src.wikientity import WikiEntity
from src.utils import convert_list


class DataReader:
    def __init__(self, config, word_2_id, attr_2_id):
        self.config = config
        self.word_2_id = word_2_id
        self.attr_2_id = attr_2_id

    def _read_data(self, data_file):
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
                pos_fw = np.minimum(pos_fw, self.config.sequence_len - 1).tolist()
                pos_bw = pos_bw[:self.config.sequence_len]
                pos_bw = np.minimum(pos_bw, self.config.sequence_len - 1).tolist()
                desc = desc[:self.config.sequence_len - 2]  # 2 for sos and eos

                if self.config.to_lower:
                    value = list(map(str.lower, value))
                    attr = list(map(str.lower, attr))
                    desc = list(map(str.lower, desc))
                desc = [self.config.sos] + desc + [self.config.eos]

                value_seq.append(convert_list(value, self.word_2_id, self.config.pad_id, self.config.unk_id))
                attr_seq.append(convert_list(attr, self.attr_2_id, self.config.pad_id, self.config.unk_id))
                pos_fw_seq.append(pos_fw)
                pos_bw_seq.append(pos_bw)
                desc_seq.append(convert_list(desc, self.word_2_id, self.config.pad_id, self.config.unk_id))

                counter += 1
                if counter % 10000 == 0:
                    print('\rprocessing file {}: {:>6d}'.format(data_file, counter), end='')
            print()

        return value_seq, attr_seq, pos_fw_seq, pos_bw_seq, desc_seq

    def read_train_data(self):
        return self._read_data(self.config.train_data)

    def read_valid_data(self):
        return self._read_data(self.config.valid_data)

    def read_test_data(self):
        return self._read_data(self.config.test_data)

import numpy as np

from src.config import Config
from src.data_reader import DataReader
from src.utils import read_dict, convert_list


def check_data(data, id_2_word, id_2_attr, config):
    value_seq, attr_seq, pos_fw_seq, pos_bw_seq, desc_seq = data

    for _ in range(5):
        print('=' * 20)
        index = np.random.randint(0, len(value_seq))
        print('id: {}'.format(index))

        value = convert_list(value_seq[index], id_2_word, config.pad, config.unk)
        attr = convert_list(attr_seq[index], id_2_attr, config.pad, config.unk)
        pos_fw = pos_fw_seq[index]
        pos_bw = pos_bw_seq[index]
        inp = [(v, a, p1, p2) for v, a, p1, p2 in zip(value, attr, pos_fw, pos_bw)]

        desc = convert_list(desc_seq[index], id_2_word, config.pad, config.unk)

        print('input: {}'.format(inp))
        print('description: {}'.format(desc))


def main():
    config = Config('.', 'temp')
    word_2_id, id_2_word = read_dict(config.word_dict)
    attr_2_id, id_2_attr = read_dict(config.attr_dict)
    data_reader = DataReader(config, word_2_id, attr_2_id)

    test_data = data_reader.read_test_data()
    check_data(test_data, id_2_word, id_2_attr, config)

    print('done')


if __name__ == '__main__':
    main()

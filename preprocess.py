import collections

from src.config import Config
from src.wikientity import WikiEntity
from src.utils import read_json_lines, save_json


def build_word_dict(config):
    cnt = 0
    word_cnt = collections.Counter()
    attr_cnt = collections.Counter()

    for line in read_json_lines(config.train_data):
        we = WikiEntity(line)

        box = we.get_box()
        for a in box.keys():
            for w in box[a].split():
                if config.to_lower:
                    w = w.lower()
                word_cnt[w] += 1
            if config.to_lower:
                a = a.lower()
            attr_cnt[a] += 1

        desc = we.get_desc()
        for w in desc.split():
            if config.to_lower:
                w = w.lower()
            word_cnt[w] += 1

        cnt += 1
        if cnt % 10000 == 0:
            print('\rprocessing: {}'.format(cnt), end='')
    print()

    word_cnt[config.pad] = attr_cnt[config.pad] = 1e9 - config.pad_id
    word_cnt[config.unk] = attr_cnt[config.unk] = 1e9 - config.unk_id
    word_cnt[config.sos] = attr_cnt[config.sos] = 1e9 - config.sos_id
    word_cnt[config.eos] = attr_cnt[config.eos] = 1e9 - config.eos_id
    word_cnt[config.num] = attr_cnt[config.num] = 1e9 - config.num_id
    word_cnt[config.time] = attr_cnt[config.time] = 1e9 - config.time_id
    print('number of words in word counter: {}'.format(len(word_cnt)))
    print('number of words in attribute counter: {}'.format(len(attr_cnt)))

    word_dict = {}
    for word, _ in word_cnt.most_common(config.word_size):
        word_dict[word] = len(word_dict)
    save_json(word_dict, config.word_dict)

    attr_dict = {}
    for attr, _ in attr_cnt.most_common(config.attr_size):
        attr_dict[attr] = len(attr_dict)
    save_json(attr_dict, config.attr_dict)


def preprocess():
    config = Config('.', 'temp')

    print('building word dict...')
    build_word_dict(config)


if __name__ == '__main__':
    preprocess()

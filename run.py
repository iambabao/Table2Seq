import os
import time
import json
import argparse
import numpy as np
import tensorflow as tf

from src.config import Config
from src.data_reader import DataReader
from src.evaluator import Evaluator
from src.model import get_model
from src.utils import read_dict, load_glove_embedding, make_batch_iter, pad_batch, convert_list

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--em', action='store_true', default=False)
parser.add_argument('--beam_search', action='store_true', default=False)
parser.add_argument('--model_file', type=str)
args = parser.parse_args()

config = Config('.', args.model, num_epoch=args.epoch, batch_size=args.batch,
                optimizer=args.optimizer, lr=args.lr,
                embedding_trainable=args.em, beam_search=args.beam_search)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def save_result(outputs, result_file, id_2_label):
    with open(result_file, 'w', encoding='utf-8') as fout:
        for line in outputs:
            res = {
                'description': ' '.join(convert_list(line, id_2_label, config.pad, config.unk))
            }
            print(json.dumps(res, ensure_ascii=False), file=fout)


def refine_outputs(outputs):
    ret = []
    for output in outputs:
        output[-1] = config.eos_id
        ret.append(output[:output.index(config.eos_id)])

    return ret


def inference(sess, model, batch_iter, verbose=True):
    outputs = []
    for step, batch in enumerate(batch_iter):
        value_seq, attr_seq, pos_fw_seq, pos_bw_seq, _ = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in value_seq])

        value_seq = np.array(pad_batch(value_seq, config.pad_id))
        attr_seq = np.array(pad_batch(attr_seq, config.pad_id))
        pos_fw_seq = np.array(pad_batch(pos_fw_seq, config.pad_id))
        pos_bw_seq = np.array(pad_batch(pos_bw_seq, config.pad_id))

        predicted_ids = sess.run(
            model.predicted_ids,
            feed_dict={
                model.value_inp: value_seq,
                model.attr_inp: attr_seq,
                model.pos_fw_inp: pos_fw_seq,
                model.pos_bw_inp: pos_bw_seq,
                model.src_len: src_len_seq,
                model.training: False
            }
        )
        outputs.extend(refine_outputs(predicted_ids.tolist()))

        if verbose:
            print('\rprocessing batch: {:>6d}'.format(step + 1), end='')
    print()

    return outputs


def run_epoch(sess, model, batch_iter, summary_writer, training, verbose=True):
    steps = 0
    total_loss = 0.0
    total_accu = 0.0
    start_time = time.time()
    for batch in batch_iter:
        value_seq, attr_seq, pos_fw_seq, pos_bw_seq, desc_seq = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in value_seq])
        tgt_len_seq = np.array([len(tgt) for tgt in desc_seq])

        value_seq = np.array(pad_batch(value_seq, config.pad_id))
        attr_seq = np.array(pad_batch(attr_seq, config.pad_id))
        pos_fw_seq = np.array(pad_batch(pos_fw_seq, config.pad_id))
        pos_bw_seq = np.array(pad_batch(pos_bw_seq, config.pad_id))
        desc_seq = np.array(pad_batch(desc_seq, config.pad_id))

        if training:
            _, loss, accu, global_step, summary = sess.run(
                [model.train_op, model.loss, model.accu, model.global_step, model.summary],
                feed_dict={
                    model.value_inp: value_seq,
                    model.attr_inp: attr_seq,
                    model.pos_fw_inp: pos_fw_seq,
                    model.pos_bw_inp: pos_bw_seq,
                    model.desc_inp: desc_seq[:, :-1],  # 1 for eos
                    model.desc_out: desc_seq[:, 1:],  # 1 for sos
                    model.src_len: src_len_seq,
                    model.tgt_len: tgt_len_seq - 1,  # 1 for eos
                    model.training: training
                }
            )
        else:
            loss, accu, global_step, summary = sess.run(
                [model.loss, model.accu, model.global_step, model.summary],
                feed_dict={
                    model.value_inp: value_seq,
                    model.attr_inp: attr_seq,
                    model.pos_fw_inp: pos_fw_seq,
                    model.pos_bw_inp: pos_bw_seq,
                    model.desc_inp: desc_seq[:, :-1],  # 1 for eos
                    model.desc_out: desc_seq[:, 1:],  # 1 for sos
                    model.src_len: src_len_seq,
                    model.tgt_len: tgt_len_seq - 1,  # 1 for eos
                    model.training: training
                }
            )

        steps += 1
        total_loss += loss
        total_accu += accu
        if steps % 10 == 0:
            summary_writer.add_summary(summary, global_step)

            if verbose:
                current_time = time.time()
                time_per_batch = (current_time - start_time) / 10.0
                start_time = current_time
                print('\rAfter {:>6d} batch(es), loss is {:>.4f}, accuracy is {:>.4f}, {:>.4f}s/batch'.format(
                    steps, loss, accu, time_per_batch), end='')
    print()

    return total_loss / steps, total_accu / steps


def train():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.train_log_dir):
        os.makedirs(config.train_log_dir)
    if not os.path.exists(config.valid_log_dir):
        os.makedirs(config.valid_log_dir)

    print('loading data...')
    word_2_id, id_2_word = read_dict(config.word_dict)
    attr_2_id, id_2_attr = read_dict(config.attr_dict)
    config.word_size = len(word_2_id)
    config.attr_size = len(attr_2_id)

    if os.path.exists(config.glove_file):
        print('loading embedding matrix from file: {}'.format(config.glove_file))
        embedding_matrix, config.word_em_size = load_glove_embedding(config.glove_file, list(word_2_id.keys()))
        print('shape of embedding matrix: {}'.format(embedding_matrix.shape))
    else:
        embedding_matrix = None

    data_reader = DataReader(config, word_2_id, attr_2_id)
    train_data = data_reader.read_train_data()
    valid_data = data_reader.read_valid_data()

    print('building model...')
    model = get_model(config, embedding_matrix)

    print('==========  Trainable Variables  ==========')
    for v in tf.trainable_variables():
        print(v)

    print('========== Gradients  ==========')
    for g in model.gradients:
        print(g)

    flag = 0
    best_loss = 1e5
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session(config=sess_config) as sess:
        if tf.train.latest_checkpoint(config.result_dir):
            saver.restore(sess, tf.train.latest_checkpoint(config.result_dir))
            print('loading model from {}'.format(tf.train.latest_checkpoint(config.result_dir)))
        else:
            tf.global_variables_initializer().run()
            print('initializing from scratch.')

        train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
        valid_writer = tf.summary.FileWriter(config.valid_log_dir, sess.graph)

        for i in range(config.num_epoch):
            print('==========  Epoch {} Train  =========='.format(i + 1))
            train_batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True)
            train_loss, train_accu = run_epoch(sess, model, train_batch_iter, train_writer, training=True, verbose=True)
            print('The average train loss is {:>.4f}, average accuracy is {:>.4f}'.format(train_loss, train_accu))

            print('==========  Epoch {} Valid  =========='.format(i + 1))
            valid_batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=False)
            valid_loss, valid_accu = run_epoch(sess, model, valid_batch_iter, valid_writer, training=False, verbose=True)
            print('The average valid loss is {:>.4f}, average accuracy is {:>.4f}'.format(valid_loss, valid_accu))

            # after 15 epoch, stop training when the loss on validation set do not decrease for 5 epoch
            if valid_loss < best_loss:
                flag = 0
                best_loss = valid_loss
            elif flag > 5:
                break
            elif i + 1 > 15:
                flag += 1
            saver.save(sess, config.model_file, global_step=i+1)
    print('done')


def test():
    assert os.path.exists(config.result_dir)

    print('loading data...')
    word_2_id, id_2_word = read_dict(config.word_dict)
    attr_2_id, id_2_attr = read_dict(config.attr_dict)
    config.word_size = len(word_2_id)
    config.attr_size = len(attr_2_id)

    if os.path.exists(config.glove_file):
        with open(config.glove_file, 'r', encoding='utf-8') as fin:
            line = fin.readline()
            config.word_em_size = len(line.strip().split()) - 1

    data_reader = DataReader(config, word_2_id, attr_2_id)
    test_data = data_reader.read_test_data()
    evaluator = Evaluator('description')

    print('building model...')
    model = get_model(config)

    saver = tf.train.Saver()
    with tf.Session(config=sess_config) as sess:
        model_file = args.model_file
        if model_file is None:
            model_file = tf.train.latest_checkpoint(config.result_dir)
        if model_file:
            saver.restore(sess, model_file)
            print('loading model from {}'.format(model_file))

            test_batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False)
            outputs = inference(sess, model, test_batch_iter, verbose=True)

            print('==========  Saving Result  ==========')
            save_result(outputs, config.test_result, id_2_word)
            evaluator.evaluate(config.test_data, config.test_result, config.to_lower)
        else:
            print('model not found')
    print('done')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        assert False

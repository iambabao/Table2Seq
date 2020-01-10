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
from src.utils import read_dict, load_glove_embedding, make_batch_iter, pad_batch, convert_list, print_title

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--do_train', action='store_true', default=False)
parser.add_argument('--do_eval', action='store_true', default=False)
parser.add_argument('--do_test', action='store_true', default=False)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--em', action='store_true', default=False)
parser.add_argument('--model_file', type=str)
parser.add_argument('--log_steps', type=int, default=10)
parser.add_argument('--save_steps', type=int, default=100)
parser.add_argument('--pre_train_epochs', type=int, default=4)
parser.add_argument('--early_stop', action='store_true', default=False)
parser.add_argument('--beam_search', action='store_true', default=False)
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


def run_test(sess, model, test_data, verbose=True):
    outputs = []
    batch_iter = make_batch_iter(list(zip(*test_data)), config.batch_size, shuffle=False, verbose=verbose)
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


def run_evaluate(sess, model, valid_data, summary_writer=None, verbose=True):
    steps = 0
    total_loss = 0.0
    total_accu = 0.0
    batch_iter = make_batch_iter(list(zip(*valid_data)), config.batch_size, shuffle=True, verbose=verbose)
    for batch in batch_iter:
        value_seq, attr_seq, pos_fw_seq, pos_bw_seq, desc_seq = list(zip(*batch))
        src_len_seq = np.array([len(src) for src in value_seq])
        tgt_len_seq = np.array([len(tgt) for tgt in desc_seq])

        value_seq = np.array(pad_batch(value_seq, config.pad_id))
        attr_seq = np.array(pad_batch(attr_seq, config.pad_id))
        pos_fw_seq = np.array(pad_batch(pos_fw_seq, config.pad_id))
        pos_bw_seq = np.array(pad_batch(pos_bw_seq, config.pad_id))
        desc_seq = np.array(pad_batch(desc_seq, config.pad_id))

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
                model.training: False
            }
        )

        steps += 1
        total_loss += loss
        total_accu += accu
        if verbose:
            print('\rprocessing batch: {:>6d}'.format(steps + 1), end='')
        if steps % args.log_steps == 0 and summary_writer is not None:
            summary_writer.add_summary(summary, global_step)
    print()

    return total_loss / steps, total_accu / steps


def run_train(sess, model, train_data, valid_data, saver,
              train_summary_writer=None, valid_summary_writer=None, verbose=True):
    flag = 0
    loss_log = 1e9
    for i in range(config.num_epoch):
        print_title('Train Epoch: {}'.format(i + 1))
        steps = 0
        total_loss = 0.0
        total_accu = 0.0
        batch_iter = make_batch_iter(list(zip(*train_data)), config.batch_size, shuffle=True, verbose=verbose)
        for batch in batch_iter:
            start_time = time.time()
            value_seq, attr_seq, pos_fw_seq, pos_bw_seq, desc_seq = list(zip(*batch))
            src_len_seq = np.array([len(src) for src in value_seq])
            tgt_len_seq = np.array([len(tgt) for tgt in desc_seq])

            value_seq = np.array(pad_batch(value_seq, config.pad_id))
            attr_seq = np.array(pad_batch(attr_seq, config.pad_id))
            pos_fw_seq = np.array(pad_batch(pos_fw_seq, config.pad_id))
            pos_bw_seq = np.array(pad_batch(pos_bw_seq, config.pad_id))
            desc_seq = np.array(pad_batch(desc_seq, config.pad_id))

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
                    model.training: True
                }
            )

            steps += 1
            total_loss += loss
            total_accu += accu
            if verbose:
                print('\rafter {:>6d} batch(s), train loss is {:>.4f}, train accuracy is {:>.4f}, {:>.4f}s/batch'
                      .format(steps, loss, accu, time.time() - start_time), end='')
            if steps % args.log_steps == 0 and train_summary_writer is not None:
                train_summary_writer.add_summary(summary, global_step)
            if global_step % args.save_steps == 0:
                saver.save(sess, config.model_file, global_step=global_step)
                if i + 1 > args.pre_train_epochs:
                    valid_loss, valid_accu = run_evaluate(sess, model, valid_data, valid_summary_writer, verbose=False)
                    print_title('Valid Result', sep='*')
                    print('average valid loss: {:>.4f}, average valid accuracy: {:>.4f}'.format(valid_loss, valid_accu))

                    # early stop training when the loss on validation set do not decrease for more than 5 epochs
                    if valid_loss < loss_log:
                        flag = 0
                        loss_log = valid_loss
                    elif flag < 5:
                        flag += 1
                    elif args.early_stop:
                        return
        print()
        print_title('Train Result', sep='*')
        print('average train loss: {:>.4f}, average train accuracy: {:>.4f}'.format(
            total_loss / steps, total_accu / steps))
    saver.save(sess, config.model_file, global_step=global_step)


def main():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.train_log_dir):
        os.makedirs(config.train_log_dir)
    if not os.path.exists(config.valid_log_dir):
        os.makedirs(config.valid_log_dir)

    print('preparing data...')
    word_2_id, id_2_word = read_dict(config.word_dict)
    attr_2_id, id_2_attr = read_dict(config.attr_dict)
    config.word_size = len(word_2_id)
    config.attr_size = len(attr_2_id)

    embedding_matrix = None
    if args.do_train:
        if os.path.exists(config.glove_file):
            print('loading embedding matrix from file: {}'.format(config.glove_file))
            embedding_matrix, config.word_em_size = load_glove_embedding(config.glove_file, list(word_2_id.keys()))
            print('shape of embedding matrix: {}'.format(embedding_matrix.shape))
    else:
        if os.path.exists(config.glove_file):
            with open(config.glove_file, 'r', encoding='utf-8') as fin:
                line = fin.readline()
                config.word_em_size = len(line.strip().split()) - 1

    data_reader = DataReader(config, word_2_id, attr_2_id)
    evaluator = Evaluator('description')

    print('building model...')
    model = get_model(config, embedding_matrix)
    saver = tf.train.Saver(max_to_keep=10)

    if args.do_train:
        print('loading data...')
        train_data = data_reader.read_train_data()
        valid_data = data_reader.read_valid_data()

        print_title('Trainable Variables')
        for v in tf.trainable_variables():
            print(v)

        print_title('Gradients')
        for g in model.gradients:
            print(g)

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}'.format(model_file))
                saver.restore(sess, model_file)
            else:
                print('initializing from scratch.')
                tf.global_variables_initializer().run()

            train_writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
            valid_writer = tf.summary.FileWriter(config.valid_log_dir, sess.graph)

            run_train(sess, model, train_data, valid_data, saver, train_writer, valid_writer, verbose=True)

    if args.do_eval:
        print('loading data...')
        valid_data = data_reader.read_valid_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}'.format(model_file))
                saver.restore(sess, model_file)

                valid_loss, valid_accu = run_evaluate(sess, model, valid_data, summary_writer=None, verbose=True)
                print('average valid loss: {:>.4f}, average valid accuracy: {:>.4f}'.format(valid_loss, valid_accu))
            else:
                print('model not found')

    if args.do_test:
        print('loading data...')
        test_data = data_reader.read_test_data()

        with tf.Session(config=sess_config) as sess:
            model_file = args.model_file
            if model_file is None:
                model_file = tf.train.latest_checkpoint(config.result_dir)
            if model_file is not None:
                print('loading model from {}'.format(model_file))
                saver.restore(sess, model_file)

                outputs = run_test(sess, model, test_data, verbose=True)

                print_title('Saving Result')
                save_result(outputs, config.test_result, id_2_word)
                evaluator.evaluate(config.test_data, config.test_result, config.to_lower)
            else:
                print('model not found')


if __name__ == '__main__':
    main()
    print('done')

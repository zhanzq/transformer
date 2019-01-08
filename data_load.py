# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np
import codecs
import regex
from prepro import clean_data
import os


def load_vocab(hp):
    vocab = [line.split()[0] for line in codecs.open(hp.vocab_path, 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    if len(vocab) > hp.max_voc_sz:
        vocab = vocab[:hp.max_voc_sz]
    print('min word frequency : %d' % hp.min_cnt)
    print('filtered vocab size : %d' % len(vocab))

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(hp, source_sents, target_sents):
    word2idx, idx2word = load_vocab(hp)

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [word2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        y = [word2idx.get(word, 1) for word in (target_sent + u" </S>").split()]

        # truncate
        x = x[:hp.maxlen]
        y = y[:hp.maxlen]

        x_list.append(np.array(x))
        y_list.append(np.array(y))
        Sources.append(source_sent)
        Targets.append(target_sent)

        # drop directly
        # if max(len(x), len(y)) <=hp.maxlen:
        #     x_list.append(np.array(x))
        #     y_list.append(np.array(y))
        #     Sources.append(source_sent)
        #     Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_data(hp, data_path):
    s_sents, t_sents = [], []
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        line_num = 0
        s_sen = fin.readline()
        while s_sen != '':
            t_sen = fin.readline()
            s_sen = clean_data(s_sen)
            t_sen = clean_data(t_sen)
            if s_sen and t_sen:
                s_sents.append(s_sen)
                t_sents.append(t_sen)
            line_num += 2
            # if line_num % 100 == 0:
            #     print('read %d lines' % line_num)
            s_sen = fin.readline()

    X, Y, Sources, Targets = create_data(hp, s_sents, t_sents)

    return X, Y, Sources, Targets


def get_batch_data(hp):
    # Load train data
    X, Y, Sources, Targets = load_data(hp, hp.train_path)

    # calc total batch count
    num_batch = (len(X) + hp.batch_size - 1) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # # Create Queues
    # dataset = tf.data.Dataset.from_tensor_slices(tuple([X, Y]))\
    #     .shuffle(buffer_size=hp.batch_size*100)\
    #     .batch(batch_size=hp.batch_size)\
    #     # .repeat(count=hp.num_epochs)
    # input_queues = dataset.make_one_shot_iterator()
    # x, y = input_queues.get_next()

    # create batch queues
    input_queues = tf.train.slice_input_producer([X, Y])
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size,
                                  capacity=hp.batch_size * 64,
                                  min_after_dequeue=hp.batch_size * 32,
                                  allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()


def queue_test(X, Y):
    # Create Queues
    batch_size = 4
    num_epochs = 6
    dataset = tf.data.Dataset.from_tensor_slices(tuple([X, Y])) \
        .shuffle(buffer_size=batch_size * 100). \
        batch(batch_size=batch_size). \
        repeat(count=num_epochs)
    input_queues = dataset.make_one_shot_iterator()
    x, y = input_queues.get_next()

    return x, y


def main():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    Y = [-1, -2, -3, -4, -5, -6, -7, -8, -9]

    x, y = queue_test(X, Y)
    with tf.Session() as sess:
        while True:
            _x, _y = sess.run([x, y])
            print(_x, _y)


if __name__ == '__main__':
    main()
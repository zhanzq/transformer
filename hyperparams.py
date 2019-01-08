# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

import os
import json


class Hyperparams:
    '''Hyperparameters'''

    def __init__(self, conf_path):
        dct = None
        with open(conf_path, 'r') as fin:
            dct = json.load(fin)

        # default set
        self.project_dir = '/Users/zhanzq/github/transformer/'
        self.dataset = 'lenovo'
        self.data_dir = '/Users/zhanzq/github/transformer/corpora'
        # data
        self.train_path = '/Users/zhanzq/github/transformer/corpora/lenovo/train.txt'
        self.test_path = '/Users/zhanzq/github/transformer/corpora/lenovo/test.txt'
        self.vocab_path = '/Users/zhanzq/github/dct/lenovo/vocab.tsv'
        self.res_path = '/Users/zhanzq/github/transformer/results/lenovo/res.txt'
        self.logdir = '/Users/zhanzq/github/transformer/logdir/'  # log directory
        self.max_voc_sz = 10000

        # training
        self.batch_size = 32  # alias = N
        self.lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.

        # model
        self.maxlen = 25  # Maximum number of words in a sentence. alias = T.
        # Feel free to increase this if you are ambitious.
        self.min_cnt = 10  # words whose occurred less than min_cnt are encoded as <UNK>.
        self.hidden_units = 512  # alias = C
        self.num_blocks = 6  # number of encoder/decoder blocks
        self.num_epochs = 60
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.sinusoid = False  # If True, use sinusoid. If false, positional embedding.

        if dct is not None:
            self.project_dir = dct["project_dir"]
            self.dataset = dct["dataset"]
            self.data_dir = dct["data_dir"]
            self.train_path = dct["train_path"]
            self.test_path = dct["test_path"]
            self.vocab_path = dct["vocab_path"]
            self.res_path = dct["res_path"]
            self.logdir = dct["logdir"]
            self.max_voc_sz = dct["max_voc_sz"]
            self.batch_size = dct["batch_size"]
            self.lr = dct["lr"]
            self.maxlen = dct["maxlen"]
            self.min_cnt = dct["min_cnt"]
            self.hidden_units = dct["hidden_units"]
            self.num_blocks = dct["num_blocks"]
            self.num_epochs = dct["num_epochs"]
            self.num_heads = dct["num_heads"]
            self.dropout_rate = dct["dropout_rate"]
            self.sinusoid = dct["sinusoid"]


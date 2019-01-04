# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/zhanzq/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter
import sys

def clean_data(text):
    text = regex.sub('[^,\.\?!\'":a-zA-Z]', ' ', text)
    text = regex.sub(',', ' , ', text)  # ',' ==> ' ,'
    text = regex.sub('\.', ' . ', text)  # '.' ==> ' !'
    text = regex.sub('\?', ' ? ', text)  # '?' ==> ' ?'
    text = regex.sub('!', ' ! ', text)  # '?' ==> ' ?'
    text = regex.sub('\'', ' \'', text)
    text = regex.sub('"', ' " ', text)
    text = regex.sub(':', ' : ', text)
    text = regex.sub(' +', ' ', text.strip())

    return text


def make_vocab(hp, fpath):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''
    words = []
    word2cnt = None
    if type(fpath) is str:
        text = codecs.open(fpath, 'r', 'utf-8').read()
        text = clean_data(text)
        # text = regex.sub("[^\s\p{Latin}']", " ", text)  # equals '[^\sA-Za-z]', Latin chars contains 26+26 characters
        words = text.split()
    else:
        for _fpath in fpath:
            text = codecs.open(_fpath, 'r', 'utf-8').read()
            text = clean_data(text)
            # text = regex.sub("[^\s\p{Latin}']", " ", text)
            words.extend(text.split())
    word2cnt = Counter(words)

    vocab_dir = os.path.dirname(hp.vocab_path)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    with codecs.open(hp.vocab_path, 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    conf_dir = './conf/'
    dataset = 'lenovo'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    hp = Hyperparams(os.path.join(conf_dir, '%s.conf' % dataset))
    make_vocab(hp, [hp.train_path, hp.test_path])
    print("Done")
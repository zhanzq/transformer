# -*- coding: utf-8 -*-
# /usr/bin/python2
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
import csv


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = clean_data(line[3])
      text_b = clean_data(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = line[0]
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


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
        fnames = []
        for _fpath in fpath:
            if os.path.isfile(_fpath):
                fnames.append(_fpath)
            else:
                for fname in os.listdir(_fpath):
                    fnames.append(os.path.join(_fpath, fname))

        for _fpath in fnames:
            text = codecs.open(_fpath, 'r', 'utf-8').read()
            text = clean_data(text)
            # text = regex.sub("[^\s\p{Latin}']", " ", text)
            words.extend(text.split())
    word2cnt = Counter(words)

    vocab_dir = os.path.dirname(hp.vocab_path)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    with codecs.open(hp.vocab_path, 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    conf_dir = './conf/'
    dataset = 'trec'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    hp = Hyperparams(os.path.join(conf_dir, '%s.conf' % dataset))
    make_vocab(hp, [hp.train_path, hp.valid_path, hp.test_path])
    print("Done")
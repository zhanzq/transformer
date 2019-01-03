# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np
import sys

from hyperparams import Hyperparams
from data_load import load_data, load_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

def eval(hp):
    # Load graph
    g = Graph(hp, is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Y, Sources, Targets = load_data(hp, hp.test_path)
    word2idx, idx2word = load_vocab(hp)
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            res_dir = os.path.dirname(hp.res_path)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            with codecs.open(hp.res_path, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()

                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))
                                          
if __name__ == '__main__':
    conf_dir = './conf/'
    dataset = 'lenovo'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    hp = Hyperparams(os.path.join(conf_dir, '%s.conf' % dataset))

    eval(hp)
    print("Done")
    
    
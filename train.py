# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by zhanzq.
zhanzhqiang09@126.com
https://www.github.com/zhanzq/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams
from data_load import get_batch_data, load_vocab
from modules import *
import os, codecs
from tqdm import tqdm
import sys

class Graph():
    def __init__(self, hp, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data(hp) # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

            # Load vocabulary    
            word2idx, idx2word = load_vocab(hp)

            ## Embedding
            self.enc = embedding(self.x,
                                 vocab_size=len(word2idx),
                                 num_units=hp.hidden_units,
                                 scale=True,
                                 scope="embed")

            # use the same dictionary
            ## Embedding
            self.dec = embedding(self.decoder_inputs,
                                 vocab_size=len(word2idx),
                                 num_units=hp.hidden_units,
                                 scale=True,
                                 scope="embed",
                                 reuse=True)

            # Encoder
            with tf.variable_scope("encoder"):

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                 
                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])


            # Decoder
            with tf.variable_scope("decoder"):

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                
            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(word2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(word2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # # Load vocabulary
    # word2idx, idx2word = load_vocab()

    conf_dir = './conf/'
    dataset = 'lenovo'
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    hp = Hyperparams(os.path.join(conf_dir, '%s.conf' % dataset))

    # Construct graph
    g = Graph(hp, "train")
    print("Graph loaded")
    
    # Start session
    # sv = tf.train.MonitoredTrainingSession(master=g.graph, checkpoint_dir=hp.logdir, save_checkpoint_secs=0)
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    epoch = 1
    with sv.managed_session() as sess:
        # while not sv.should_stop():
        #     for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
        #         _, batch_loss = sess.run([g.train_op, g.mean_loss])
        # gs = sess.run(g.global_step)
        # sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
        # epoch += 1

        for epoch in range(1, hp.num_epochs+1):
            if sv.should_stop():
                break
            g.num_batch = 2
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, batch_loss = sess.run([g.train_op, g.mean_loss])

            gs = sess.run(g.global_step)
            sv.saver.save(sess, os.path.join(hp.logdir, 'model_epoch_%02d_gs_%d' % (epoch, gs)))
    
    print("Done")
    


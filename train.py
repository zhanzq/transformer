# -*- coding: utf-8 -*-
# /usr/bin/python2
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
                self.x, self.y, self.num_batch = get_batch_data(hp)  # (N, T)
            else:  # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None))

            # Load vocabulary
            word2idx, idx2word = load_vocab(hp)

            ## Embedding
            with tf.variable_scope("embedding_layer"):
                self.enc = embedding(self.x,
                                     vocab_size=len(word2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")
                else:
                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))


            # Encoder
            with tf.variable_scope("encoder"):

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
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

                ## pooler
                with tf.variable_scope("pooler"):
                    # We "pool" the model by simply taking the hidden state corresponding
                    # to the first token. We assume that this has been pre-trained
                    first_token_tensor = tf.squeeze(self.enc[:, 0:1, :], axis=1)
                    self.output_layer = tf.layers.dense(
                        first_token_tensor,
                        hp.hidden_units,
                        activation=tf.tanh,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_weights = tf.get_variable(
                "output_weights", [hp.num_labels, hp.hidden_units],
                initializer=tf.truncated_normal_initializer(stddev=0.02))   ## initializer_range

            output_bias = tf.get_variable(
                "output_bias", [hp.num_labels], initializer=tf.zeros_initializer())

            with tf.variable_scope("loss"):
                if is_training:
                    # I.e., 0.1 dropout
                    self.output_layer = tf.nn.dropout(self.output_layer, keep_prob=0.9)

                logits = tf.matmul(self.output_layer, output_weights, transpose_b=True)
                self.logits = tf.nn.bias_add(logits, output_bias)

                self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))

                probabilities = tf.nn.softmax(logits, axis=-1)
                log_probs = tf.nn.log_softmax(logits, axis=-1)

                one_hot_labels = tf.one_hot(self.y, depth=hp.num_labels, dtype=tf.float32)

                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)

            with tf.variable_scope("accuracy"):
                self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.preds, self.y)))
                tf.summary.scalar('acc', self.acc)

            if is_training:
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('batch_loss', self.loss)
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

        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop():
                break
            # g.num_batch = 8
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                _, batch_loss = sess.run([g.train_op, g.loss])

            gs = sess.run(g.global_step)
            sv.saver.save(sess, os.path.join(hp.logdir, 'model_epoch_%02d_gs_%d' % (epoch, gs)))

    print("Done")



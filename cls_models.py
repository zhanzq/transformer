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
import copy
import json
import collections
import re

import six


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      print("not found %s" % name)
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    for transformer
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, keep_prob=1.0 - dropout_prob)
  return output


def layer_norm(input_tensor, name=None):
  """
  Run layer normalization on the last dimension of the tensor.
  for bert
  """
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  # return tf.contrib.layers.xavier_initializer()                   # transformer initializer
  return tf.truncated_normal_initializer(stddev=initializer_range)  # bert initializer


def multihead_attention(queries,
                        keys,
                        config,
                        num_units=None,
                        num_heads=8,
                        hidden_dropout_rate=0.1,
                        attention_dropout_rate=0.1,
                        is_training=True,
                        causality=False,
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope("self", reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, name="query", activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, name="key", activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, name="value", activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=attention_dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    with tf.variable_scope("output"):
        # Residual connection
        attention_output = tf.layers.dense(
            outputs,
            config.hidden_size,
            kernel_initializer=create_initializer(config.initializer_range))
        attention_output = dropout(attention_output, hidden_dropout_rate)
        attention_output = layer_norm(attention_output + queries)

    return attention_output


def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def encoder_layer(config, inputs, is_training):
    ## Blocks
    all_layer_outputs = []
    encode_output = inputs
    for layer_idx in range(config.num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            with tf.name_scope("self_attention"):
                # self attention
                with tf.variable_scope("attention"):
                    encode_output = multihead_attention(queries=encode_output,
                                                        keys=encode_output,
                                                        config=config,
                                                        num_units=config.hidden_size,
                                                        num_heads=config.num_attention_heads,
                                                        hidden_dropout_rate=config.hidden_dropout_prob,
                                                        attention_dropout_rate=config.attention_probs_dropout_prob,
                                                        is_training=is_training,
                                                        causality=False)

                with tf.name_scope("feed_forward"):
                    # The activation is only applied to the "intermediate" hidden layer.
                    with tf.variable_scope("intermediate"):
                        intermediate_output = tf.layers.dense(
                                                            encode_output,
                                                            config.intermediate_size,
                                                            activation=get_activation(config.hidden_act),
                                                            kernel_initializer=create_initializer(config.initializer_range))

                    # Down-project back to `hidden_size` then add the residual.
                    with tf.variable_scope("output"):
                        layer_output = tf.layers.dense(
                                                            intermediate_output,
                                                            config.hidden_size,
                                                            kernel_initializer=create_initializer(config.initializer_range))
                        layer_output = dropout(layer_output, config.hidden_dropout_prob)
                        layer_output += encode_output
                        layer_output = layer_norm(layer_output)
                        prev_layer_output = layer_output
                        all_layer_outputs.append(layer_output)

    return all_layer_outputs


def decoder_layer(config, encode_output, decode_output, is_training):
    ## Blocks
    all_layer_outputs = []
    for layer_idx in range(config.num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            ### Multihead Attention
            # self attention
            with tf.variable_scope("self_attention"):
                decode_output = multihead_attention(queries=decode_output,
                                                    keys=decode_output,
                                                    num_units=config.hidden_units,
                                                    num_heads=config.num_heads,
                                                    hidden_dropout_rate=config.hidden_dropout_prob,
                                                    attention_dropout_rate=config.attention_probs_dropout_prob,
                                                    is_training=is_training,
                                                    causality=False)

            with tf.variable_scope("vanilla_attention"):
                decode_output = multihead_attention(queries=decode_output,
                                                    keys=encode_output,
                                                    num_units=config.hidden_units,
                                                    num_heads=config.num_heads,
                                                    hidden_dropout_rate=config.hidden_dropout_prob,
                                                    attention_dropout_rate=config.attention_probs_dropout_prob,
                                                    is_training=is_training,
                                                    causality=False)
            with tf.variable_scope("feed_forward"):
                # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                                                        decode_output,
                                                        config.intermediate_size,
                                                        activation=config.intermediate_act_fn,
                                                        kernel_initializer=create_initializer(config.initializer_range))

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    layer_output = tf.layers.dense(
                                                        intermediate_output,
                                                        config.hidden_size,
                                                        kernel_initializer=create_initializer(config.initializer_range))
                layer_output = dropout(layer_output, config.hidden_dropout_prob)
                layer_output += decode_output
                layer_output = layer_norm(layer_output)
                decode_output = layer_output
                all_layer_outputs.append(layer_output)

    return all_layer_outputs


def embedding_layer(config, input_ids, token_type_ids, reuse=None):
    embedding_output, embedding_table = embedding(
                                                    input_ids=input_ids,
                                                    vocab_size=config.vocab_size,
                                                    embedding_size=config.hidden_size,
                                                    zero_pad=False,
                                                    scale=False,
                                                    initializer_range=config.initializer_range,
                                                    word_embedding_name="word_embeddings",
                                                    reuse=reuse)

    # Add positional embeddings and token type embeddings, then layer
    # normalize and perform dropout.
    embedding_output = embedding_postprocessor(
                                                    input_tensor=embedding_output,
                                                    use_token_type=True,
                                                    token_type_ids=token_type_ids,
                                                    token_type_vocab_size=config.type_vocab_size,
                                                    token_type_embedding_name="token_type_embeddings",
                                                    use_position_embeddings=True,
                                                    position_embedding_name="position_embeddings",
                                                    initializer_range=config.initializer_range,
                                                    max_position_embeddings=config.max_position_embeddings,
                                                    dropout_prob=config.hidden_dropout_prob)

    return embedding_output, embedding_table


def embedding(input_ids,
              vocab_size,
              embedding_size=128,
              zero_pad=False,
              scale=False,
              initializer_range=0.02,
              word_embedding_name="word_embeddings",
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.name_scope("embeddings_table"):
        lookup_table = tf.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range))
        # initializer=tf.contrib.layers.xavier_initializer()
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, embedding_size]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, input_ids)

        if scale:   # normalize the word embedding
            outputs = outputs * (embedding_size ** 0.5)

    return outputs, lookup_table


"""
not checked
"""
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = input_tensor.get_shape().as_list()
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [-1, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm(output)
  output = dropout(output, dropout_prob)
  return output


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Transformer(object):
    def __init__(self, config, max_seq_len=128, num_labels=None, learning_rate=1.0e-5, is_training=True):
        self.is_training = is_training
        self.inputs = {}
        self.inputs["input_ids"]   = tf.placeholder(tf.int32, shape=[None, max_seq_len])
        self.inputs["input_mask"]  = tf.placeholder(tf.int32, shape=[None, max_seq_len])
        self.inputs["segment_ids"] = tf.placeholder(tf.int32, shape=[None, max_seq_len])
        self.inputs["label_ids"]   = tf.placeholder(tf.int32, shape=[None])
        self.num_labels = num_labels
        self.build_model(config, is_training, self.inputs, learning_rate=learning_rate)

    def build_model(self, config, is_training, inputs, learning_rate=1.0e-5):
        input_ids   = inputs["input_ids"]
        input_mask  = inputs["input_mask"]
        segment_ids = inputs["segment_ids"]
        label_ids   = inputs["label_ids"]

        with tf.variable_scope("bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                embedding_x, self.embedding_table = embedding_layer(config=config,
                                                                         input_ids=input_ids,
                                                                         token_type_ids=segment_ids)

            with tf.variable_scope("encoder"):
                all_encoder_layers = encoder_layer(config=config,
                                                inputs=embedding_x,
                                                is_training= is_training)

            self.sequence_output = all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                                                first_token_tensor,
                                                config.hidden_size,
                                                activation=tf.tanh,
                                                kernel_initializer=create_initializer(config.initializer_range))

            self.pooled_output = dropout(self.pooled_output, dropout_prob=config.hidden_dropout_prob)
            with tf.variable_scope("accuracy"):
                # Final linear projection
                output_weights = tf.get_variable(
                    "output_weights", [self.num_labels, config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                output_bias = tf.get_variable(
                    "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

                self.logits = tf.matmul(self.pooled_output, output_weights, transpose_b=True)
                self.logits = tf.nn.bias_add(self.logits, output_bias)
                self.preds    = tf.to_int32(tf.argmax(self.logits, axis=-1))
                self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.preds, label_ids)))

                tf.summary.scalar('acc', self.accuracy)

            if is_training:
                # Loss
                with tf.variable_scope("loss"):
                    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                labels=label_ids)
                    self.loss = tf.reduce_mean(self.loss)

                    # Training Scheme
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    self.optimizer   = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
                    self.train_op    = self.optimizer.minimize(self.loss, global_step=self.global_step)

                    # Summary
                    tf.summary.scalar('loss', self.loss)
                    self.merged = tf.summary.merge_all()

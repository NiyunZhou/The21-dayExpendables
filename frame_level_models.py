# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

import tensorflow.contrib.layers as tl  # BN

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "LogisticModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

# parameter by Zhouny
flags.DEFINE_integer("segments_num", 3, "Number of segments before feed into model")
flags.DEFINE_integer("max_frames", 300, "Max frames used for processing")
flags.DEFINE_string("temporal_pooling", "max_pooling", "Pooling strategy for temporal pooling")
flags.DEFINE_integer("pooling_k_size", 3, "Kernel size of pooling")
flags.DEFINE_integer("pooling_stride", 3, "Stride of pooling")
flags.DEFINE_float("drop_prob", 0.5, "Drop out probability before FC")


class LstmModel(models.BaseModel):
    def chop_frames(self, model_input, start_frame, length):
        model_input = tf.slice(model_input, [0, 0, start_frame, 0], [-1,-1, length, -1])
        return model_input

    def add_weighted_lstm_output(self, lstm_output):
        N = lstm_output.shape[1].value
        weight = 1.0 / N
        sum = tf.multiply(tf.slice(lstm_output, [0, 0, 0], [-1, 1, -1]) , tf.constant(weight))
        for n in xrange(1,N):
            weight = (n + 1.0) / N
            sum = tf.add(sum, tf.multiply(tf.slice(lstm_output, [0, n, 0], [-1, 1, -1]) , tf.constant(weight)))

        return sum



    def create_model(self, model_input, vocab_size, num_frames, is_training, **unused_params):
        #TODO: the value of max_frames - num_frames ?
        """Creates a model which uses a stack of LSTMs to represent the video.
        shape(model_input) = [batch_size 300 1024]  
        shape(num_frames) = [batch_size]  eg: num_frames[176 215 183 148 122 140 300 153 183 185 176 300 178 183 300 148 300 268 215 145 222 172 230 300 166 150 268 122 161 122 122 122]
        vocab_size == 4716
    
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
    
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """

        # calculate how many frames in each segments
        frames_each_seg = FLAGS.max_frames / FLAGS.segments_num
        segments = []

        # reshape like a 2D image
        model_input = tf.expand_dims(model_input, 1) # [batch_size 1 300 1024]

        # build the model (according to 2017 TS-LSTM and Temporal-Inception: Exploiting Spatiotemporal Dynamics for Activity Recognition )
        # separate tensor
        for s in xrange(FLAGS.segments_num):
            segments.append(self.chop_frames(model_input, s * frames_each_seg, frames_each_seg ))

        # BN + pooling
        for s in xrange(FLAGS.segments_num):
            segments[s] = tl.batch_norm(segments[s],center=True, scale=True, is_training=is_training)
            segments[s] = tf.nn.max_pool(segments[s], [1, 1, FLAGS.pooling_k_size, 1], [1, 1, FLAGS.pooling_stride, 1], padding="VALID")

        # Concatinate
        concat_seg = tf.concat([segments[s] for s in xrange(FLAGS.segments_num)], axis=2)

        # build LSTM model
        lstm_size = FLAGS.lstm_cells  # 1024
        number_of_layers = FLAGS.lstm_layers  # 2

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
            ])

        loss = 0.0

        # reshape tensor to 3 dim
        shape = concat_seg.shape
        concat_seg = tf.reshape(concat_seg, [-1, shape[2].value, shape[3].value])

        # calculate sequence length
        num_frames = tf.cast(num_frames, tf.float32)
        sequence_length = tf.ceil((num_frames - FLAGS.pooling_k_size + 1.0) / FLAGS.pooling_stride) # size after pooling
        sequence_length = tf.cast(sequence_length, tf.int32)

        # feed into LSTM
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, concat_seg,
                                           sequence_length=sequence_length,
                                           dtype=tf.float32)

        # BN
        fc_input = tl.batch_norm(state[-1].h,center=True, scale=True, is_training=is_training)

        # dropout
        fc_input = tf.nn.dropout(fc_input, FLAGS.drop_prob)

        # set linear model
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=fc_input,
            vocab_size=vocab_size,
            **unused_params)


class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)



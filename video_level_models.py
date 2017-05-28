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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_float("drop_prob", 0.5, "Drop out probability before FC")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, is_training, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features. 1024 + 128 = 1152
      vocab_size: The number of classes in the dataset. 

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    model_input = slim.batch_norm(
        model_input,
        center=True,
        scale=True,
        is_training=is_training)
    fc1_out = slim.fully_connected(model_input, 9216)
    fc1_out = slim.dropout(fc1_out, FLAGS.drop_prob, is_training=is_training)
    fc2_out = slim.fully_connected(fc1_out, 4608)
    fc2_out = slim.dropout(fc2_out, FLAGS.drop_prob,  is_training=is_training)
    fc3_out = slim.fully_connected(fc2_out, 1152)
    fc3_out = slim.batch_norm(
        fc3_out,
        center=True,
        scale=True,
        is_training=is_training)
    fc4_in = tf.add(model_input, fc3_out)

    fc4_in = slim.dropout(fc4_in, FLAGS.drop_prob,  is_training=is_training)
    fc4_out = slim.fully_connected(fc4_in, 9216)
    fc4_out = slim.dropout(fc4_out, FLAGS.drop_prob,  is_training=is_training)
    output = slim.fully_connected(fc4_out, vocab_size, activation_fn=tf.nn.sigmoid,
                                   weights_regularizer=slim.l2_regularizer(l2_penalty))

    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class SkipModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    layer_1 = slim.fully_connected(
        model_input, 1152, scope='fc/fc_1')
    layer_2 = slim.fully_connected(
        model_input + layer_1, 1152, scope='fc/fc_2')
    layer_3 = slim.fully_connected(
        layer_2, 1152, scope='fc/fc_3')
    output = slim.fully_connected(
        model_input + layer_2 + layer_3, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope='fc/fc_4')
    return {"predictions": output}

class DenseModel(models.BaseModel):

  def dense_block(self, model_input,is_training):
    layer_1 = slim.fully_connected(
          model_input, 2304, scope='fc1/fc_1')
    layer_1 = slim.dropout(layer_1, 0.7, is_training=is_training)
    layer_1 = slim.batch_norm(layer_1, center=True, scale=True, is_training=is_training)
    layer_2 = slim.fully_connected(
          layer_1, 2304, scope='fc1/fc_2')
    layer_2 = slim.dropout(layer_2, 0.6, is_training=is_training)
    layer_2 = slim.batch_norm(layer_2, center=True, scale=True, is_training=is_training)
    layer_3 = slim.fully_connected(
          layer_1 + layer_2, 2304, scope='fc1/fc_3')
    layer_3 = slim.dropout(layer_3, 0.5, is_training=is_training)
    layer_3 = slim.batch_norm(layer_3, center=True, scale=True, is_training=is_training)
    output_1 = slim.fully_connected(
          layer_1 + layer_2 + layer_3, 2304, scope='fc1/fc_4')
    output_1 = slim.dropout(output_1, 0.5, is_training=is_training)
    output_1 = slim.batch_norm(output_1, center=True, scale=True, is_training=is_training)
    return output_1

  def create_model(self, model_input, vocab_size, is_training, l2_penalty=1e-8, **unused_params):
    model_input = slim.batch_norm(model_input, center=True, scale=True, is_training=is_training)
    output_2 = self.dense_block(model_input, is_training)
    output_2 = slim.fully_connected(
        output_2, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty), scope='fc2/fc_4')
    return {"predictions": output_2}

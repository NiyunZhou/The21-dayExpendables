class MlpMoeModel(models.BaseModel):
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

    layer_1 = slim.fully_connected(
        model_input, 1152, scope='fc/fc_1')
    layer_2 = slim.fully_connected(
        layer_1, 2304, scope='fc/fc_2')
    expert_activations = slim.fully_connected(
        layer_2,
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
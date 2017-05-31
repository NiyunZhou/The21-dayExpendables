class FrameLevelStdModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    #batch_size = model_input.get_shape().as_list()[0]
    #print model_input.get_shape()
    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    #print denominators.get_shape()
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators
    #print avg_pooled.get_shape()
    expand_pooled = tf.tile(tf.expand_dims(avg_pooled,1),[1,max_frames,1])
    # expand_max_frames = tf.expand_dims(max_frames,0)
    #batch = denominators.get_shape().as_list()[0]
    # expand_max_frames = tf.tile(tf.expand_dims(expand_max_frames,0),[batch,feature_size])
    # print expand_max_frames.get_shape()
    #print expand_pooled.get_shape()
    # expand_max_frames = tf.cast(expand_max_frames,tf.float32)
    #zero_frames =  - denominators
    #print zero_frames.get_shape()
    deviation = model_input - expand_pooled
    square_deviation = tf.square(deviation)
    sum_square_deviation = tf.reduce_sum(square_deviation,axis=[1])
    #print square_deviation.get_shape()
    square_pooled = tf.square(avg_pooled)
    max_frames = tf.cast(max_frames,tf.float32)
    error = square_pooled * max_frames - multiply(square_pooled,denominators) 
    
    real_sum_square = sum_square_deviation - error
    std = tf.sqrt(real_sum_square/denominators)
    new_input = tf.concat([avg_pooled,std],1)
    #print new_input.get_shape()
    output = slim.fully_connected(
        new_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

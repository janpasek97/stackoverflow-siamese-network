"""
Attention layer implementation
"""
import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """
    Implementation of attention layer based on article:
    'Bidirectional LSTM with attention mechanism and convolutional layer for text classification'
    by Gang Liu and Jiabao Guo
    """
    def __init__(self, size, return_attention=False, **kwargs):
        """
        Constructor
        :param size: size of the attention learned vector
        :param return_attention: True/False whether the layer shall return attention weights together with the processed input
        :param kwargs: tf.keras.layers.Layer kwargs
        """
        self.init = tf.initializers.get('uniform')
        self.supports_masking = True
        self.size = size
        self.return_attention = return_attention
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Weight initialization
        :param input_shape: shape of input tensor
        :return: void
        """
        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]
        assert len(input_shape) == 3
        self.w = self.add_weight(shape=(input_shape[2], self.size),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.b = self.add_weight(shape=[self.size],
                                 name='{}_b'.format(self.name),
                                 initializer=self.init)
        self.v = self.add_weight(shape=[self.size],
                                 name='{}_v'.format(self.name),
                                 initializer=self.init)

        super(Attention, self).build(input_shape)

    def call(self, input, mask=None):
        """
        Layer forward pass implementation. Implementation is based on the article. See class doc.

        :param input: input tensor
        :param mask: sequence mask
        :return: (output tensor) or (output tensor, attention weights), depends on return_attention parameter in the constructor
        """
        u = tf.tanh(tf.add(tf.matmul(input, self.w), self.b))
        logits = tf.tensordot(u, self.v, axes=(2, 0))
        alpha = tf.math.exp(logits)

        # masked timesteps have zero weight
        if mask is not None:
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            alpha = alpha * mask

        alpha = alpha / tf.keras.backend.sum(alpha, axis=1, keepdims=True)
        out = tf.keras.backend.sum(input * tf.expand_dims(alpha, axis=2), axis=1)
        if self.return_attention:
            return [out, alpha]
        return out

    def get_output_shape_for(self, input_shape):
        """
        Compute output shape based on the knowledge of input shape
        :param input_shape: shape of input tensor
        :return: output tensor shape
        """
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute output shape based on the knowledge of the input shape and return_attention attribute
        :param input_shape: shape of input tensor
        :return: output tensor shape
        """
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, input, input_mask=None):
        """
        Transorm input sequence mask to the output mask
        :param input: input tensor
        :param input_mask: input mask
        :return: output mask
        """
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None
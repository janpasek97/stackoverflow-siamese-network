import tensorflow as tf
import numpy as np
from network.utils import constants


class SNLIModel(tf.keras.Model):
    def __init__(self, embedding_weights_pretrained, num_reserved_words):
        super(SNLIModel, self).__init__()

        pretrained_embedding_matrix = tf.Variable(embedding_weights_pretrained, trainable=False)
        oov_embedding_matrix = tf.Variable(np.random.rand(num_reserved_words + 1, embedding_weights_pretrained.shape[1]), trainable=True)
        embedding_matrix = tf.concat([oov_embedding_matrix, pretrained_embedding_matrix], axis=0)

        self.embedding = tf.keras.layers.Embedding(embedding_weights_pretrained.shape[0] + num_reserved_words + 1,
                                                   embedding_weights_pretrained.shape[1],
                                                   mask_zero=True, weights=[embedding_matrix])

        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dropout2 = tf.keras.layers.Dropout(0.3)

        self.lstm1 = tf.keras.layers.LSTM(300, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        self.lstm2 = tf.keras.layers.LSTM(300, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)

        self.dense1 = tf.keras.layers.Dense(600, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.05))
        self.dense2 = tf.keras.layers.Dense(600, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.05))
        self.out = tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.05))

    def __call__(self, x, **kwargs):
        x1, x2 = tf.unstack(x, axis=0)
        x1, x2 = self.embedding(x1), self.embedding(x2)
        x1, x2 = self.lstm1(x1), self.lstm1(x2)
        x1, x2 = self.lstm2(x1), self.lstm2(x2)

        concat = tf.concat([x1, x2], axis=1)

        xc = self.dense1(concat)
        xc = self.dropout1(xc)
        xc = self.dense2(xc)
        xc = self.dropout2(xc)
        return self.out(xc)

import tensorflow as tf
import numpy as np
from network.utils import constants


class BiDirLSTM2LDense2LConcat3Cls(tf.keras.Model):
    def __init__(self, embedding_weights_pretrained, hparams, num_reserved_words):
        super(BiDirLSTM2LDense2LConcat3Cls, self).__init__()

        pretrained_embedding_matrix = tf.Variable(embedding_weights_pretrained, trainable=False)
        oov_embedding_matrix = tf.Variable(np.random.rand(num_reserved_words + 1, embedding_weights_pretrained.shape[1]), trainable=True)
        embedding_matrix = tf.concat([oov_embedding_matrix, pretrained_embedding_matrix], axis=0)

        self.embedding = tf.keras.layers.Embedding(embedding_weights_pretrained.shape[0] + num_reserved_words + 1,
                                                   embedding_weights_pretrained.shape[1],
                                                   mask_zero=True, weights=[embedding_matrix])
        self.dropout1 = tf.keras.layers.Dropout(hparams[constants.HP_FIRST_DROPOUT])
        self.dropout2 = tf.keras.layers.Dropout(hparams[constants.HP_SECOND_DROPOUT])
        self.bi_dir_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))
        self.bi_dir_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))
        self.dense1 = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(hparams[constants.HP_L2_REGULARIZATION]))
        self.dense2 = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(hparams[constants.HP_L2_REGULARIZATION]))
        self.out = tf.keras.layers.Dense(3, activation="softmax", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(hparams[constants.HP_L2_REGULARIZATION]))

    def __call__(self, x, **kwargs):
        x1, x2 = tf.unstack(x, axis=0)
        x1, x2 = self.embedding(x1), self.embedding(x2)
        x1, x2 = self.dropout1(x1), self.dropout1(x2)
        x1, x2 = self.bi_dir_lstm1(x1), self.bi_dir_lstm1(x2)
        x1, x2 = self.dropout2(x1), self.dropout2(x2)
        x1, x2 = self.bi_dir_lstm2(x1), self.bi_dir_lstm2(x2)
        x1, x2 = self.dropout2(x1), self.dropout2(x2)
        concat = tf.concat([x1, x2], axis=1)
        xc = self.dense1(concat)
        xc = self.dropout2(xc)
        xc = self.dense2(xc)
        xc = self.dropout2(xc)
        return self.out(xc)

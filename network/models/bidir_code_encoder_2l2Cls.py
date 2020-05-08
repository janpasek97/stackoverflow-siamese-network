import tensorflow as tf
import numpy as np
from network.utils import constants
from network.layers.network_layers_lstm_attention import Attention


class BiDirCodeEncoder2L2Cls(tf.keras.Model):
    def __init__(self, embedding_weights_pretrained, hparams, num_reserved_words, code_embeddings_weights_pretrained):
        super(BiDirCodeEncoder2L2Cls, self).__init__()

        # setup text embedding
        pretrained_embedding_matrix = tf.Variable(embedding_weights_pretrained, trainable=False)
        oov_embedding_matrix = tf.Variable(
            np.random.rand(num_reserved_words + 1, embedding_weights_pretrained.shape[1]),
            trainable=True)
        embedding_matrix = tf.concat([oov_embedding_matrix, pretrained_embedding_matrix], axis=0)
        self.embedding = tf.keras.layers.Embedding(embedding_weights_pretrained.shape[0] + num_reserved_words + 1,
                                                   embedding_weights_pretrained.shape[1],
                                                   mask_zero=True, weights=[embedding_matrix])

        # setup code embedding
        pretrained_code_embedding_matrix = tf.Variable(code_embeddings_weights_pretrained, trainable=False)
        oov_code_embedding_matrix = tf.Variable(np.random.rand(2, code_embeddings_weights_pretrained.shape[1]),
                                                trainable=True)
        code_embedding_matrix = tf.concat([oov_code_embedding_matrix, pretrained_code_embedding_matrix], axis=0)
        self.code_embedding = tf.keras.layers.Embedding(code_embeddings_weights_pretrained.shape[0] + 2,
                                                        code_embeddings_weights_pretrained.shape[1],
                                                        mask_zero=True, weights=[code_embedding_matrix])

        # dropouts
        self.dropout1 = tf.keras.layers.Dropout(hparams[constants.HP_FIRST_DROPOUT])
        self.dropout2 = tf.keras.layers.Dropout(hparams[constants.HP_SECOND_DROPOUT])

        # code encoder part
        self.code_bidir_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))
        self.code_bidir_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))

        # text encoder
        self.bi_dir_lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))
        self.bi_dir_lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False))

        # output layers
        self.dense1 = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(
            hparams[constants.HP_L2_REGULARIZATION]))
        self.dense2 = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(
            hparams[constants.HP_L2_REGULARIZATION]))
        self.out = tf.keras.layers.Dense(2, activation="softmax", use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(
                                             hparams[constants.HP_L2_REGULARIZATION]))

    def __call__(self, x, training=False, **kwargs):
        text_x1, text_x2, code_x1, code_x2 = tf.unstack(x, axis=0)

        # text encoder
        text_x1, text_x2 = self.embedding(text_x1), self.embedding(text_x2)
        text_x1, text_x2 = self.dropout1(text_x1, training=training), self.dropout1(text_x2, training=training)
        text_x1, text_x2 = self.bi_dir_lstm1(text_x1), self.bi_dir_lstm1(text_x2)
        text_x1, text_x2 = self.dropout2(text_x1, training=training), self.dropout2(text_x2, training=training)
        text_x1, text_x2 = self.bi_dir_lstm2(text_x1), self.bi_dir_lstm2(text_x2)
        text_x1, text_x2 = self.dropout2(text_x1, training=training), self.dropout2(text_x2, training=training)

        # code encoder
        code_x1, code_x2 = self.code_embedding(code_x1), self.code_embedding(code_x2)
        code_x1, code_x2 = self.dropout1(code_x1, training=training), self.dropout1(code_x2, training=training)
        code_x1, code_x2 = self.code_bidir_lstm1(code_x1), self.code_bidir_lstm1(code_x2)
        code_x1, code_x2 = self.dropout2(code_x1, training=training), self.dropout2(code_x2, training=training)
        code_x1, code_x2 = self.code_bidir_lstm2(code_x1), self.code_bidir_lstm2(code_x2)
        code_x1, code_x2 = self.dropout2(code_x1, training=training), self.dropout2(code_x2, training=training)

        # features concatenation
        x1 = tf.concat([text_x1, code_x1], axis=1)
        x2 = tf.concat([text_x2, code_x2], axis=1)
        concat = tf.concat([x1, x2], axis=1)

        # output decision layers
        xc = self.dense1(concat)
        xc = self.dropout2(xc, training=training)
        xc = self.dense2(xc)
        return self.out(xc)

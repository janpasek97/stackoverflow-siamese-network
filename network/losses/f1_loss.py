"""
F1 loss implementation
"""
import tensorflow as tf


class F1Loss(tf.keras.losses.Loss):
    """
    F1 loss implementation based on online article: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    """
    def __init__(self, num_classes):
        """
        Constructor
        :param num_classes: number of classes into which the examples are classified
        """
        super(F1Loss, self).__init__()
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
        Computes value of loss based on predicted and true labels.
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: void of f1 loss on given batch
        """
        y_true = tf.squeeze(tf.one_hot(tf.keras.backend.cast(y_true, tf.int32), self.num_classes, dtype=tf.float32), axis=1)
        tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=0)
        tn = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
        fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + tf.keras.backend.epsilon())
        r = tp / (tp + fn + tf.keras.backend.epsilon())

        f1 = 2 * p * r / (p + r + tf.keras.backend.epsilon())
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
        return 1 - tf.keras.backend.mean(f1)

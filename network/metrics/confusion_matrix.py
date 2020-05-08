"""
Confusion matrix metric implementation
"""
import tensorflow as tf


class ConfusionMatrix(tf.keras.metrics.Metric):
    """
    Confusion matrix implementation as a child class from tf.keras.metrics.Metric
    """
    def __init__(self, num_classes, average=None, name='conf_matrix'):
        """
        Constructor
        :param num_classes: number of possible classes
        :param average: not used at the moment
        :param name: name of the metric
        """
        super(ConfusionMatrix, self).__init__(name=name)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name,
                                                shape=[self.num_classes, self.num_classes],
                                                initializer='zeros',
                                                dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric variables after the given batch
        :param y_true: true labels
        :param y_pred: predicted labels
        :param sample_weight:
        :return:void
        """
        predictions = tf.argmax(y_pred, axis=1)
        self.confusion_matrix.assign_add(tf.math.confusion_matrix(labels=y_true, predictions=predictions))

    def result(self):
        """
        return resulted confusion matrix
        :return: resulted confusion matrix
        """
        return self.confusion_matrix

    def reset_states(self):
        """
        reset the variables
        :return: void
        """
        self.confusion_matrix.assign(tf.zeros([self.num_classes, self.num_classes], tf.int32))
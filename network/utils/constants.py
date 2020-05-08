"""
Model configuration and important constants definitions
"""
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from network.models.bidir_lstm_1l_diff_3cls import BiDirLSTM1L3Cls
from network.models.bidir_lstm_2l_diff_3cls import BiDirLSTM2L3Cls
from network.models.bidir_lstm_2l_concat_3cls import BiDirLSTM2LConcat3Cls
from network.models.bidir_lstm_1l_concat_3cls import BiDirLSTM1LConcat3Cls
from network.models.bidir_lstm_1l_concat_2cls import BiDirLSTM1LConcat2Cls
from network.models.bidir_lstm_1l_diff_2cls import BiDirLSTM1L2Cls
from network.models.bidir_lstm_2l_concat_2cls import BiDirLSTM2LConcat2Cls
from network.models.bidir_lstm_2l_diff_2cls import BiDirLSTM2L2Cls
from network.models.bidir_lstm_2l_dense_2l_concat_2cls import BiDirLSTM2LDense2LConcat2Cls
from network.models.bidir_lstm_2l_dense_2l_concat_3cls import BiDirLSTM2LDense2LConcat3Cls

from network.models.bidir_code_encoder_2l2Cls import BiDirCodeEncoder2L2Cls
from network.models.bidir_code_encoder_2l3Cls import BiDirCodeEncoder2L3Cls

from network.models.word_sum_2cls import WordSum2Cls
from network.models.word_sum_3cls import WordSum3Cls
from network.models.word_sum_concat_2cls import WordSumConcat2Cls
from network.models.word_sum_concat_3cls import WordSumConcat3Cls
from network.models.snli_model import SNLIModel

from network.losses.f1_loss import F1Loss

from network.utils.train_utils import EMBEDDING_MATRIX_PRETRAINED_FN, EMBEDDING_MATRIX_CUSTOM_TEXT_FN, \
    EMBEDDING_MATRIX_SNLI_FN
from network.utils.train_utils import WORD_TO_IDX_CUSTOM_TEXT_FN, WORD_TO_IDX_PRETRAINED_FN, WORD_TO_IDX_SNLI_FN
from network.utils.train_utils import IDX_TO_WORD_CUSTOM_TEXT_FN, IDX_TO_WORD_PRETRAINED_FN, IDX_TO_WORD_SNLI_FN

from network.utils.text_cleaner import NUM_RESERVED_WORDS

# possible models dictionary -> value is a tuple (Model class, number of classes, code embedding T/F)
ALL_MODELS_CONFIG = {"BiDirLSTM1LConcat3Cls": (BiDirLSTM1LConcat3Cls, 3, False),
                     "BiDirLSTM1L3Cls": (BiDirLSTM1L3Cls, 3, False),
                     "BiDirLSTM2LConcat3Cls": (BiDirLSTM2LConcat3Cls, 3, False),
                     "BiDirLSTM2L3Cls": (BiDirLSTM2L3Cls, 3, False),
                     "BiDirLSTM1LConcat2Cls": (BiDirLSTM1LConcat2Cls, 2, False),
                     "BiDirLSTM1L2Cls": (BiDirLSTM1L2Cls, 2, False),
                     "BiDirLSTM2LConcat2Cls": (BiDirLSTM2LConcat2Cls, 2, False),
                     "BiDirLSTM2L2Cls": (BiDirLSTM2L2Cls, 2, False),
                     "BiDirLSTM2LDense2LConcat2Cls": (BiDirLSTM2LDense2LConcat2Cls, 2, False),
                     "BiDirLSTM2LDense2LConcat3Cls": (BiDirLSTM2LDense2LConcat3Cls, 3, False),
                     "BiDirCodeEncoder2L2Cls": (BiDirCodeEncoder2L2Cls, 2, True),
                     "BiDirCodeEncoder2L3Cls": (BiDirCodeEncoder2L3Cls, 3, True),
                     "WordSum2Cls": (WordSum2Cls, 2, False),
                     "WordSum3Cls": (WordSum3Cls, 3, False),
                     "WordSumConcat3Cls": (WordSumConcat3Cls, 3, False),
                     "WordSumConcat2Cls": (WordSumConcat2Cls, 2, False),
                     "SNLI": (SNLIModel, 3, False)}

# all possible losses
LOSSES = {"f1_loss": F1Loss,
          "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy}

# all available text embeddings
# values are tuple (embedding file name, word2idx dict file name, idx2word dict file name, number of reserved words)
TEXT_EMBEDDINGS = {
    "pretrained": (EMBEDDING_MATRIX_PRETRAINED_FN, WORD_TO_IDX_PRETRAINED_FN, IDX_TO_WORD_PRETRAINED_FN,
                   NUM_RESERVED_WORDS),
    "custom": (EMBEDDING_MATRIX_CUSTOM_TEXT_FN, WORD_TO_IDX_CUSTOM_TEXT_FN, IDX_TO_WORD_CUSTOM_TEXT_FN,
               1),
    "snli": (EMBEDDING_MATRIX_SNLI_FN, WORD_TO_IDX_SNLI_FN, IDX_TO_WORD_SNLI_FN,
             NUM_RESERVED_WORDS)
}

# hyperparameters logging configuration
HP_L2_REGULARIZATION = hp.HParam('L2_regularization', hp.RealInterval(0.01, 0.1))
HP_FIRST_DROPOUT = hp.HParam('first_dropout', hp.RealInterval(0.1, 0.6))
HP_SECOND_DROPOUT = hp.HParam('second_dropout', hp.RealInterval(0.1, 0.6))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.IntInterval(64, 512))
HP_SEQUENCE_LENGTH = hp.HParam('max_sequence_length', hp.IntInterval(50, 300))
HP_MODEL = hp.HParam('model', hp.Discrete(ALL_MODELS_CONFIG.keys()))
HP_EMBEDDING = hp.HParam('embedding', hp.Discrete(TEXT_EMBEDDINGS.keys()))
HP_LOSS = hp.HParam('loss', hp.Discrete(list(LOSSES.keys())))

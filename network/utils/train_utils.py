"""
Utility functions and path constants for model training
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Path constants pointing to embedding matrices, translation dictionaries and dataset exports
WORD_TO_IDX_PRETRAINED_FN = "network/assets/pretrained_w2v/word_to_idx.dat"
IDX_TO_WORD_PRETRAINED_FN = "network/assets/pretrained_w2v/idx_to_word.dat"
EMBEDDING_MATRIX_PRETRAINED_FN = "network/assets/pretrained_w2v/embedding_matrix.npy"
WORD_TO_IDX_SNLI_FN = "network/assets/snli_w2v/word_to_idx.dat"
IDX_TO_WORD_SNLI_FN = "network/assets/snli_w2v/idx_to_word.dat"
EMBEDDING_MATRIX_SNLI_FN = "network/assets/snli_w2v/embedding_matrix.npy"
WORD_TO_IDX_CUSTOM_TEXT_FN = "network/assets/custom_w2v_text/word_to_idx.dat"
IDX_TO_WORD_CUSTOM_TEXT_FN = "network/assets/custom_w2v_text/idx_to_word.dat"
EMBEDDING_MATRIX_CUSTOM_TEXT_FN = "network/assets/custom_w2v_text/embedding_matrix.npy"
WORD_TO_IDX_CUSTOM_CODE_FN = "network/assets/custom_w2v_code/word_to_idx.dat"
IDX_TO_WORD_CUSTOM_CODE_FN = "network/assets/custom_w2v_code/idx_to_word.dat"
EMBEDDING_MATRIX_CUSTOM_CODE_FN = "network/assets/custom_w2v_code/embedding_matrix.npy"
TRAIN_DS_TEXT_FN = "network/assets/dataset_export/train_ds_text.csv"
TEST_DS_TEXT_FN = "network/assets/dataset_export/test_ds_text.csv"
DEV_DS_TEXT_FN = "network/assets/dataset_export/dev_ds_text.csv"
TRAIN_DS_CODE_FN = "network/assets/dataset_export/train_ds_code.csv"
TEST_DS_CODE_FN = "network/assets/dataset_export/test_ds_code.csv"
DEV_DS_CODE_FN = "network/assets/dataset_export/dev_ds_code.csv"
SNLI_DS_DIR = "network/assets/tf_datasets"


def plot_graphs(history, string, title, path):
    """
    Plot graphs of selected training measurement such as accuracy, f1 score or loss from the training history data
    :param history: training history data
    :param string: the measurement to be plotted
    :param title: title of the resulted graph
    :param path: file to store the graph as .png
    :return: void
    """
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("epochs")
    plt.ylabel(string)
    plt.title(title)
    plt.legend([string, "val_" + string])
    plt.savefig(path)
    plt.clf()


def load_word2idx(fname):
    """
    Load word to vocab. index translation dictionary from the given file using the pickle lib
    :param fname: pickle file with translation dictionary
    :return: python dictionary for translation words to indexes
    """
    with open(fname, "rb") as word2idx_file:
        return pickle.load(word2idx_file)


def load_idx2word(fname):
    """
    Load vocab. index to word translation dictionary from the given file using the pickle lib
    :param fname: pickle file with translation dictionary
    :return: python dictionary for translation indexes to words
    """
    with open(fname, "rb") as idx2word_file:
        return pickle.load(idx2word_file)


def load_word2vec_embeddings(fname):
    """
    Load word2vec embedding matrix from the given numpy file
    :param fname: numpy file with the embedding matrix
    :return: embedding matrix as a np.array
    """
    return np.load(fname)

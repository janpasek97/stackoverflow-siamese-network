"""
Create embedding matrices and dictionaries from Gensim W2V model
"""
from gensim.models import KeyedVectors
import pickle
import numpy as np
import argparse
from network.utils.train_utils import EMBEDDING_MATRIX_CUSTOM_TEXT_FN, IDX_TO_WORD_CUSTOM_TEXT_FN, \
    WORD_TO_IDX_CUSTOM_TEXT_FN, EMBEDDING_MATRIX_CUSTOM_CODE_FN, IDX_TO_WORD_CUSTOM_CODE_FN, WORD_TO_IDX_CUSTOM_CODE_FN

# argument enumeration definition
TYPE_CODE = "code"
TYPE_TEXT = "text"


def create_dictionaries_and_embedding(arguments):
    """
    Reads in Gensim keyed-values and creates and embedding matrix, word2idx and idx2word dictionaries thereof.
    The function takes an argument "code"/"text" to select which keyed-values to read in - whether the one for
    code or for text.

    :param arguments: "code"/"text" to select if dictionaries and embeddings shall be made for code or for text
    :return: void
    """
    # according to selected embedding type, configure inputs and outputs
    if arguments.type == TYPE_TEXT:
        kv_file = "word2vec/wordvectors_text.kv"
        embedding_out_fn = EMBEDDING_MATRIX_CUSTOM_TEXT_FN
        word2idx_out_fn = WORD_TO_IDX_CUSTOM_TEXT_FN
        idx2word_out_fn = IDX_TO_WORD_CUSTOM_TEXT_FN
    else:
        kv_file = "word2vec/wordvectors_code.kv"
        embedding_out_fn = EMBEDDING_MATRIX_CUSTOM_CODE_FN
        word2idx_out_fn = WORD_TO_IDX_CUSTOM_CODE_FN
        idx2word_out_fn = IDX_TO_WORD_CUSTOM_CODE_FN

    # load keyed vectors from Gensim Word2Vec model and created the word2idx/idx2word dicts
    wv = KeyedVectors.load(kv_file)
    word2idx = dict((word, index + 1) for index, word in enumerate(wv.index2word))
    idx2word = dict((index + 1, word) for index, word in enumerate(wv.index2word))

    # there shall be a reserved OOV token in the dictionaries
    word2idx["<OOV>"] = 0
    idx2word[0] = "<OOV>"

    # assemble the embedding matrix from keyed vectors
    embedding_matrix = np.zeros((len(idx2word) - 1, wv["code"].shape[0]))
    for i in range(1, len(idx2word)):
        embedding_matrix[i - 1] = wv[idx2word[i]]

    # store output files
    with open(word2idx_out_fn, "wb") as word_to_idx_file:
        pickle.dump(word2idx, word_to_idx_file)

    with open(idx2word_out_fn, "wb") as idx_to_word_file:
        pickle.dump(idx2word, idx_to_word_file)

    np.save(file=embedding_out_fn, arr=embedding_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Creates dictionaries and embeddings from custom word2vec")
    parser.add_argument("-t", "--type", required=True, choices=[TYPE_CODE, TYPE_TEXT], help="Word2Vec type.")
    args = parser.parse_args()

    create_dictionaries_and_embedding(args)

"""
Create embedding matrix and dictionary from pre-trained tf.hub Word2Vec model
"""
import argparse
import csv
import pickle
import time
import numpy as np

import tensorflow_datasets as tfds
from elasticsearch_dsl.connections import connections

from data.documents import ES_HOSTS, ES_LOGIN
from data.documents import Post
from network.utils.text_cleaner import TextCleaner
from network.utils.text_cleaner import DATETIME_REPLACEMENT, NUMBERS_REPLACEMENT, URLS_REPLACEMENT, CODE_REPLACEMENT
from network.utils.text_cleaner import NUM_RESERVED_WORDS, OOV_REPLACEMENT
from network.pretrained_word2vec import embed_one
from network.utils.train_utils import EMBEDDING_MATRIX_PRETRAINED_FN, IDX_TO_WORD_PRETRAINED_FN, WORD_TO_IDX_PRETRAINED_FN

# output definitions
EMBEDDING_DIM = 250
VOCAB_SET_EXPORT_FN = "network/assets/pretrained_w2v/vocabulary_set.dat"
CLEANED_VOCAB_EXPORT_FN = "network/assets/pretrained_w2v/vocabulary_set_clean.dat"


def get_vocabulary_set(csv_file, text_cleaner):
    """
    Collect a set of all the words that appears in the train dataset
    :param csv_file: csv dataset file with format ('post1_id', 'post2_id', label)
    :param text_cleaner: Instance of text pre-processor
    :return: set of words in the vocabulary
    """
    tokenizer = tfds.features.text.Tokenizer(reserved_tokens=[DATETIME_REPLACEMENT[1:], NUMBERS_REPLACEMENT[1:],
                                                              URLS_REPLACEMENT[1:], CODE_REPLACEMENT[1:]])
    vocab_set = set()
    last_first_post = ""

    with open(csv_file, "r") as ds:
        csv_reader = csv.reader(ds, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            first_post_id = row[0]
            second_post_id = row[1]

            if not last_first_post == first_post_id:
                first_post_text = text_cleaner.clean_text(Post.get(id=first_post_id).text)
                first_post_tokens = tokenizer.tokenize(first_post_text)
                vocab_set.update(first_post_tokens)
            last_first_post = first_post_id

            second_post_text = text_cleaner.clean_text(Post.get(id=second_post_id).text)
            second_post_tokens = tokenizer.tokenize(second_post_text)
            vocab_set.update(second_post_tokens)

            if i % 1000 == 0:
                print(f"\r    Finished {i // 1000} x 1 000 posts", end="")
    print()
    print(f"Vocabulary has {len(vocab_set)} unique words.")

    return vocab_set


def clean_vocabulary(source_vocab):
    """
    Remove words that has no pre-trained embedding from the given vocabulary.
    :param source_vocab: original 'dirty' vocalbulary
    :return: vocabulary with only the words that have an embedding vector
    """
    ignored_words = set()
    for i, w in enumerate(source_vocab):
        # try to embed the word using tf.hub pre-trained W2V
        embedded_w = embed_one(np.array([w])).numpy()
        if not np.any(embedded_w):
            ignored_words.add(w)

        if i % 1000 == 0:
            print(f"\r    Finished {i // 1000} x 1 000 words", end="")

    return source_vocab - ignored_words


def create_vocab_set_and_embeddings(args):
    """
    Create vocabulary for embedding, corresponding translation dictionaries and embedding matrix
    :param args: parsed command line arguments
    :return:void
    """
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)

    # create the vocabulary or load it
    if args.vocab_set:
        start_time = time.time()
        print("Getting vocabulary set ...")
        vocabulary = sorted(get_vocabulary_set("../dataset/ds_export.csv", TextCleaner()))
        run_time = time.time() - start_time
        print(f"    {int(run_time / 60)} min, {int(run_time % 60)} s")

        vocabulary = set(vocabulary)
        with open(VOCAB_SET_EXPORT_FN, "wb") as vocab_out:
            pickle.dump(vocabulary, vocab_out)
    else:
        try:
            with open(VOCAB_SET_EXPORT_FN, "rb") as vocab_file:
                print("Raw vocabulary set imported successfully ...")
                vocabulary = set(pickle.load(vocab_file))
                print(f"Raw vocabulary set contains {len(vocabulary)} words ...")
        except FileNotFoundError:
            print("ERROR: File with exported vocabulary set does not exist! Please create one using option -v")
            exit(-1)

    # clean the vocabulary or load the clean vocab. from file
    if args.clean_vocab:
        print("Cleaning vocabulary ...")
        start_time = time.time()
        cleaned_vocab = set(clean_vocabulary(vocabulary))
        print(f"\n  Cleaned vocabulary has {len(cleaned_vocab)} unique words.")
        run_time = time.time() - start_time
        print(f"    {int(run_time / 60)} min, {int(run_time % 60)} s")
        with open(CLEANED_VOCAB_EXPORT_FN, "wb") as cleaned_vocab_out:
            pickle.dump(cleaned_vocab, cleaned_vocab_out)
    else:
        try:
            with open(CLEANED_VOCAB_EXPORT_FN, "rb") as clean_vocab_file:
                print("Clean vocabulary set imported successfully ...")
                cleaned_vocab = set(pickle.load(clean_vocab_file))
                print(f"Clean vocabulary set contains {len(cleaned_vocab)} words ...")
        except FileNotFoundError:
            print("ERROR: File with export of cleaned dataset does not exist!. Please create one using option -c")
            exit(-1)

    # create word2idx and idx2word dictionaries
    if args.dictionaries:
        print("Creating and exporting word dictionary and inverse word dictionary ...")
        cleaned_vocab_list = sorted(cleaned_vocab)
        word_dict = dict((w, i+NUM_RESERVED_WORDS) for i, w in enumerate(cleaned_vocab_list))
        inverse_dict = dict((i+NUM_RESERVED_WORDS, w) for i, w in enumerate(cleaned_vocab_list))
        word_dict[OOV_REPLACEMENT] = 0
        word_dict[NUMBERS_REPLACEMENT] = 1
        word_dict[DATETIME_REPLACEMENT] = 2
        word_dict[URLS_REPLACEMENT] = 3
        word_dict[CODE_REPLACEMENT] = 4
        inverse_dict[0] = OOV_REPLACEMENT
        inverse_dict[1] = NUMBERS_REPLACEMENT
        inverse_dict[2] = DATETIME_REPLACEMENT
        inverse_dict[3] = URLS_REPLACEMENT
        inverse_dict[4] = CODE_REPLACEMENT

        with open(WORD_TO_IDX_PRETRAINED_FN, "wb") as word_to_idx_file:
            pickle.dump(word_dict, word_to_idx_file)

        with open(IDX_TO_WORD_PRETRAINED_FN, "wb") as idx_to_word_file:
            pickle.dump(inverse_dict, idx_to_word_file)

        print("     Export finished.")

    # create and export the embedding matrix from tf.hub W2V model
    if args.embedding:
        print("Creating embedding matrix ...")
        try:
            with open(IDX_TO_WORD_PRETRAINED_FN, "rb") as idx_to_word_file:
                idx_to_word_dict = pickle.load(idx_to_word_file)
        except FileNotFoundError:
            print("ERROR: file with inverse word dictionary does not exists! Create one using -m option")
            exit(-1)
        start_time = time.time()
        embedding_matrix = np.zeros((len(idx_to_word_dict)-NUM_RESERVED_WORDS, EMBEDDING_DIM))
        for i in range(NUM_RESERVED_WORDS, len(idx_to_word_dict)):
            embedded_word = embed_one(np.array([idx_to_word_dict[i]]))
            embedding_matrix[i-NUM_RESERVED_WORDS] = embedded_word.numpy()
        np.save(file=EMBEDDING_MATRIX_PRETRAINED_FN, arr=embedding_matrix)
        run_time = time.time() - start_time
        print(f"    {int(run_time / 60)} min, {int(run_time % 60)} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Creates vocabulary set and embeddings from the whole dataset")
    parser.add_argument("-v", "--vocab_set", action="store_true", help="Creates vocabulary set")
    parser.add_argument("-c", "--clean_vocab", action="store_true",
                        help="Cleans the vocabulary so that only known words are preserved")
    parser.add_argument("-d", "--dictionaries", action="store_true",
                        help="Creates dictionaries and inverse dictionaries for converting string to sequence of numbers")
    parser.add_argument("-e", "--embedding", action="store_true",
                        help="Creates embedding matrix for all known words in dictionaries")
    arguments = parser.parse_args()

    create_vocab_set_and_embeddings(arguments)

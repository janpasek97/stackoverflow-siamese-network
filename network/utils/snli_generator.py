"""
Implementation of the SNLI dataset generator.
"""
from network.utils.text_cleaner import TextCleaner
from network.utils.train_utils import SNLI_DS_DIR
import tensorflow_datasets as tfds


class SNLIGenerator:
    """
    Simplified implementation of dataset generator for generating the SNLI items. It is a wrapper for the
    tf.datasets SNLI ds, to make a compatible interface to the rest of the code.
    """
    PAD_END = "end"
    PAD_BEGINNING = "beginning"
    PAD_WITH = 0

    def __init__(self, word2idx, idx2word, split="train", max_sentence_length=250, pad_sentences=True, padding_type=PAD_END):
        """
        Constructor
        :param word2idx: dictionary for word to vocab. index translation
        :param idx2word: dictionary for vocab. index to word translation
        :param split: 'train', 'validation', 'test' to choose which part of the dataset to use
        :param max_sentence_length: maximum length of the sequences
        :param pad_sentences: True/False whether to pad sentences or not
        :param padding_type: type of padding - "end" or "beginning"
        """
        self._split = split
        self._pad_sentences = pad_sentences
        self._max_sentence_length = max_sentence_length
        self._padding_type = padding_type
        self._word2idx = word2idx
        self._idx2word = idx2word
        self._snli_data, snli_info = tfds.load("snli", split=split, with_info=True, data_dir=SNLI_DS_DIR)
        self._cleaner = TextCleaner()

    def generate(self):
        """
        Generator function used for generating the individual items
        :return: dataset item as a tensor
        """
        for example in self._snli_data:
            first_post_text = example['hypothesis'].numpy().decode("utf-8")
            second_post_text = example['premise'].numpy().decode("utf-8")
            label = example['label'].numpy()

            if label == -1:
                continue

            first_post_text = self._cleaner.clean_text(first_post_text).split(' ')
            second_post_text = self._cleaner.clean_text(second_post_text).split(' ')

            first_post_idxs = [self._word2idx.get(word, 0) + 1 for word in first_post_text]
            second_post_idxs = [self._word2idx.get(word, 0) + 1 for word in second_post_text]

            if self._pad_sentences:
                first_post_idxs = self._pad_sequence(first_post_idxs, self._max_sentence_length)
                second_post_idxs = self._pad_sequence(second_post_idxs, self._max_sentence_length)

            yield ((first_post_idxs, second_post_idxs), label)

    def _pad_sequence(self, idxs, max_len):
        """
        Pad sequences at the beginning or at the end with 0
        :param idxs: input sequence
        :param max_len: padding length
        :return: padded sequence
        """
        if len(idxs) < max_len:
            pad_length = max_len - len(idxs)
            padding = [SNLIGenerator.PAD_WITH] * pad_length
            if self._padding_type == SNLIGenerator.PAD_BEGINNING:
                return padding + idxs
            else:
                return idxs + padding
        else:
            return idxs[:max_len]

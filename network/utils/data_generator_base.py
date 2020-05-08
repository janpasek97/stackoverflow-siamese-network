"""
Implementation of base class for dataset generators
"""
import csv
import sys

csv.field_size_limit(sys.maxsize) # some sequences are too long, so the limit has to be extended to the max. size


class DataGeneratorBase:
    """
    Base abstract class for each dataset generator. It provides function for generating the items and padding the
    sequences.
    """
    PAD_END = "end"
    PAD_BEGINNING = "beginning"
    PAD_WITH = 0

    def __init__(self, csv_text_file, word2idx, idx2word, max_sentence_length=250, pad_sentences=True,
                 padding_type=PAD_END, use_3_cls=False, embed_code=False, csv_code_file=None, code2idx=None,
                 idx2code=None, pad_code=True, max_code_length=None):
        """
        Constructor
        :param csv_text_file: textual dataset csv file
        :param word2idx: dictionary for word to vocab. index translation
        :param idx2word: dictionary for vocab. index to word translation
        :param max_sentence_length: maximum length of the sequences
        :param pad_sentences: True/False whether to pad sentences or not
        :param padding_type: type of padding - "end" or "beginning"
        :param use_3_cls: whether to handle the dataset as 2 class (different, duplicates) or 3 class (... + similar)
        :param embed_code: whether the generator shall return code sequences as well - if yes following 3 params must be provided
        :param csv_code_file: code dataset csv file, required if embed_code == True
        :param code2idx: dictionary for code token to vocab. index translation, required if embed_code == True
        :param idx2code: dictionary for vocab. index to code token translation, required if embed_code == True
        :param pad_code: True/False whether to pad the code or not
        :param max_code_length: maximum length of the code sequences
        """
        if embed_code:
            assert code2idx is not None
            assert idx2code is not None
            assert csv_code_file is not None
            if pad_code:
                assert max_code_length is not None

        self._csv_text_file = csv_text_file
        self._pad_sentences = pad_sentences
        self._max_sentence_length = max_sentence_length
        self._padding_type = padding_type
        self._word2idx = word2idx
        self._idx2word = idx2word
        self._use_3_cls = use_3_cls
        self._embed_code = embed_code
        self._csv_code_file = csv_code_file
        self._idx2code = idx2code
        self._code2idx = code2idx
        self._pad_code = pad_code
        self._max_code_length = max_code_length

    def get_post_content_and_label(self, csv_row):
        """
        Get dataset item from the source. Must be implemented by the child class
        :param csv_row: row of the dataset csv used for generating the item
        :return: dataset item
        """
        raise NotImplementedError()

    def generate(self):
        """
        Generator function used for generating the individual items
        :return: dataset item as a tensor
        """
        # open the textual csv
        text_f = open(self._csv_text_file, "r", encoding="utf-8")
        csv_text_reader = csv.reader(text_f, delimiter=",")
        code_f = None

        # open the code csv if necessary
        if self._embed_code:
            code_f = open(self._csv_code_file, "r", encoding="utf-8")
            csv_code_reader = csv.reader(code_f, delimiter=",")

        # iterate over one or both csv files simultaneously
        while True:
            try:
                text_row = next(csv_text_reader)
            except StopIteration:
                text_f.close()
                if code_f is not None:
                    code_f.close()
                break

            if self._embed_code:
                try:
                    code_row = next(csv_code_reader)
                except StopIteration:
                    text_f.close()
                    if code_f is not None:
                        code_f.close()
                    break

            # collect the dataset item
            first_post_text, second_post_text, label = self.get_post_content_and_label(text_row)
            if self._embed_code:
                first_post_code, second_post_code, label_red = self.get_post_content_and_label(code_row)
                assert label == label_red

            # convert to 2 class if necessary
            if not self._use_3_cls:
                if label == 2 or label == 1:
                    label = 0
                else:
                    label = 1

            # tokenize text
            first_post_text = first_post_text.split(' ')
            second_post_text = second_post_text.split(' ')

            # translate text to vocab indexes
            first_post_idxs = [self._word2idx.get(word, 0) + 1 for word in first_post_text]
            second_post_idxs = [self._word2idx.get(word, 0) + 1 for word in second_post_text]

            if self._embed_code:
                # tokenize code
                first_post_code = first_post_code.split(' ')
                second_post_code = second_post_code.split(' ')

                # translate code to vocab indexes
                first_post_code_idxs = [self._code2idx.get(token, 0) + 1 for token in first_post_code]
                second_post_code_idxs = [self._code2idx.get(token, 0) + 1 for token in second_post_code]

            # pad sentences
            if self._pad_sentences:
                first_post_idxs = self._pad_sequence(first_post_idxs, self._max_sentence_length)
                second_post_idxs = self._pad_sequence(second_post_idxs, self._max_sentence_length)

            # pad code sequences
            if self._embed_code and self._pad_code:
                first_post_code_idxs = self._pad_sequence(first_post_code_idxs, self._max_code_length)
                second_post_code_idxs = self._pad_sequence(second_post_code_idxs, self._max_code_length)

            # yield a dataset item
            if self._embed_code:
                yield ((first_post_idxs, second_post_idxs, first_post_code_idxs, second_post_code_idxs), label)
            else:
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
            padding = [DataGeneratorBase.PAD_WITH] * pad_length
            if self._padding_type == DataGeneratorBase.PAD_BEGINNING:
                return padding + idxs
            else:
                return idxs + padding
        else:
            return idxs[:max_len]

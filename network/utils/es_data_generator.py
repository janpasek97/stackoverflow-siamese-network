"""
Implementation of dataset generator that loads the items from Elasticsearch directly
"""
from network.utils.data_generator_base import DataGeneratorBase
from data.documents import Post


class ESDataGenerator(DataGeneratorBase):
    """
    Elasticsearch dataset generator. Child class of DataGeneratorBase
    """
    def __init__(self, csv_text_file, word2idx, idx2word, max_sentence_length=250, pad_sentences=True,
                 padding_type=DataGeneratorBase.PAD_END, use_3_cls=False, embed_code=False, csv_code_file=None,
                 code2idx=None, idx2code=None, pad_code=True, max_code_length=None):
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
        super(ESDataGenerator, self).__init__(csv_text_file, word2idx, idx2word, max_sentence_length, pad_sentences,
                                              padding_type, use_3_cls, embed_code, csv_code_file=csv_code_file,
                                              code2idx=code2idx, idx2code=idx2code, pad_code=pad_code,
                                              max_code_length=max_code_length)

    def get_post_content_and_label(self, csv_row):
        """
        Get dataset item from the the Elasticsearch based on the provided ids
        :param csv_row: row of the dataset csv with post ids and label
        :return: dataset item
        """
        first_post_id = csv_row[0]
        second_post_id = csv_row[1]
        label = csv_row[2]

        first_post_content = Post.get(id=first_post_id).text
        second_post_content = Post.get(id=second_post_id).text

        return first_post_content, second_post_content, int(label)

"""
Text pre-processing utility class
"""
import re
from bs4 import BeautifulSoup

# regular expression definitions for the text cleaner
url_re = re.compile(r'https?://\.*[a-zA-Z./?=0-9&]*')
left_angle_bracket_re = re.compile(r'&lt;')
right_angle_bracket_re = re.compile(r'&gt;')
wrong_brackets_re = re.compile(r' < | > ')
new_lines_re = re.compile('\\n|\\r|\n|\r')
special_characters_re = re.compile(r'\s-\s|[\"\'(){\}\[\]:_<>=@\/\-$%]+|&amp')
multiple_spaces_re = re.compile(r'\s{2,}')
punctuation_re = re.compile(r'[\!\.\,\;\?]+')
dates_re = re.compile(r'\d{1,2}\. ?\d{1,2}\. ?\d{4}|\d{1,2}\/\d{1,2}\/\d{4}|\d{4}\/\d{1,2}\/\d{1,2}|\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{1,2}-\d{1,2}')
date_time_re = re.compile(r'\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?|[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1]) (2[0-3]|[01][0-9]):[0-5][0-9](:[0-5][0-9])?(.[0-9]+)?')
time_re = re.compile(r'[0-2]?[0-9]:[0-5][0-9](:[0-5][0-9])?')
numbers_re = re.compile(r'\d+(\.\d+)*')

NUM_RESERVED_WORDS = 5  # number of special reserved tokens

# special tokens
OOV_REPLACEMENT = "<OOV>"
DATETIME_REPLACEMENT = " <datetime>"
NUMBERS_REPLACEMENT = " <numbers>"
URLS_REPLACEMENT = " <url>"
CODE_REPLACEMENT = " <code>"


class TextCleaner:
    """
    Text cleaner is a text pre-processing class that provides methods for the complete pre-processing of the Stackoverflow
    data and creates prepared tokens
    """
    def __init__(self, valid_tags=[]):
        """
        Constructor
        :param valid_tags: List of HTML tags that shall be kept in the source text
        """
        self.__valid_tags = valid_tags

    def clean_text(self, text):
        """
        Complete text pre-processing
        :param text: input text
        :return: pre-processed text
        """
        res = TextCleaner.remove_new_lines(text)
        res = TextCleaner.replace_urls(res)
        res = self.__sanitize_html(res)
        res = TextCleaner.replace_date_and_time(res)
        res = TextCleaner.remove_special_characters(res)
        res = TextCleaner.replace_numbers(res)
        res = TextCleaner.replace_angle_brackets(res)
        res = TextCleaner.remove_punctuation(res)
        res = TextCleaner.remove_wrong_angle_brackets(res)
        res = TextCleaner.remove_multiple_spaces(res)
        return res.strip().lower()

    @staticmethod
    def remove_punctuation(text):
        """
        Remove punctiation from the input text
        :param text: input text
        :return: text without punctuation
        """
        return punctuation_re.sub(' ', text)

    @staticmethod
    def remove_multiple_spaces(text):
        """
        Remove all redundant multiple spaces
        :param text: input text
        :return: text without multiple neighboring spaces
        """
        return multiple_spaces_re.sub(' ', text)

    @staticmethod
    def remove_special_characters(text):
        """
        Remove special characters from the text
        :param text: input text
        :return: text without special characters
        """
        return special_characters_re.sub(' ', text)

    @staticmethod
    def replace_angle_brackets(text):
        """
        Remove escaped angle brackets from the text
        :param text: input text
        :return: text without angle brackets
        """
        res = left_angle_bracket_re.sub('<', text)
        return right_angle_bracket_re.sub('>', res)

    @staticmethod
    def remove_wrong_angle_brackets(text):
        """
        Remove escaped angle brackets from the text
        :param text: input text
        :return: text without angle brackets
        """
        return wrong_brackets_re.sub(' ', text)

    @staticmethod
    def replace_urls(text):
        """
        Replace urls with special '<url>' token
        :param text: input text
        :return: text with replaced urls
        """
        return url_re.sub(URLS_REPLACEMENT, text)

    @staticmethod
    def replace_date_and_time(text):
        """
        Replace date time information with special '<datetime>' token
        :param text: input text
        :return: text with replaced dates and time information
        """
        text = date_time_re.sub(DATETIME_REPLACEMENT, text)
        text = time_re.sub(DATETIME_REPLACEMENT, text)
        return dates_re.sub(DATETIME_REPLACEMENT, text)

    @staticmethod
    def remove_new_lines(text):
        """
        Remove new line characters from the text
        :param text: input text
        :return: text without new line characters
        """
        return new_lines_re.sub(' ', text)

    @staticmethod
    def replace_numbers(text):
        """
        Replace numbers with special '<number>' token
        :param text: input text
        :return: text with replaced numbers
        """
        return numbers_re.sub(NUMBERS_REPLACEMENT, text)

    def __sanitize_html(self, value):
        """
        Remove all HTML tags from the code and replace all '<pre><code>' areas with special token '<code>'
        :param value: input text
        :return: text without code and HTML tags
        """
        soup = BeautifulSoup(value, "lxml")

        for tag in soup.find_all("code"):
            tag.replaceWith(CODE_REPLACEMENT)

        for tag in soup.find_all(True):
            if tag.name not in self.__valid_tags:
                tag.hidden = True

        return str(soup)

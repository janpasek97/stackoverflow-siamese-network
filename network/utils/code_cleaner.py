"""
Code preprocessing implementation
"""
import re
from bs4 import BeautifulSoup

# regular expression definitions for the text cleaner
code_comment_re = re.compile(r"(#[^\n]*\n)|(//[^\n]*\n)|(/\*[\s]*\*/)")
float_re = re.compile(r"\d*\.\d+")
integer_re = re.compile(r"(?<!h)\d+")
escaped_special_characters = re.compile(r"\\'|\\\"")
multiple_spaces_re = re.compile(r'\s{2,}')
new_lines_re = re.compile('\\n|\\r|\n|\r')
token_chars = re.compile(r"(\(\))|([^a-zA-Z ])| |(a-zA-Z)+")

FLOAT_REPLACEMENT = " FLOAT_TOKEN "
INTEGER_REPLACEMENT = " INTEGER_TOKEN "


class CodeCleaner:
    """
    Code cleaner is a code pre-processing class that provides methods for the complete preprocessing of the Stackoverflow
    data and creates prepared tokens
    """
    def __init__(self, valid_tags=[]):
        """
        Constructor
        :param valid_tags: List of HTML tags that shall be kept in the source text
        """
        self.__valid_tags = valid_tags

    def clean_code(self, text):
        """
        Complete code pre-processing
        :param text: input text
        :return: pre-processed text
        """
        res = CodeCleaner.extract_code_from_html(text)
        res = CodeCleaner.strip_comments(res)
        res = CodeCleaner.replace_floats(res)
        res = CodeCleaner.replace_integers(res)
        res = CodeCleaner.strip_escaped_special_characters(res)
        res = CodeCleaner.strip_new_lines(res)
        res = CodeCleaner.tokenize(res)
        res = CodeCleaner.strip_multiple_spaces(res)
        return res.strip().lower()

    @staticmethod
    def extract_code_from_html(value):
        """
        Extract the code snippets from the HTML formatted post body
        :param value: input text
        :return: text with extracted code only
        """
        soup = BeautifulSoup(value, "html5lib")
        res = ""

        for tag in soup.find_all("code"):
            if tag.parent.name == "pre":
                if tag.string:
                    res += tag.string

        return str(res)

    @staticmethod
    def strip_comments(value):
        """
        Remove all coments from the code
        :param value: input code
        :return: code without comments
        """
        return code_comment_re.sub('\n', value)

    @staticmethod
    def replace_floats(value):
        """
        Replace all float numbers with special '<float>' token
        :param value: input code
        :return: code with replaced floats
        """
        return float_re.sub(FLOAT_REPLACEMENT, value)

    @staticmethod
    def replace_integers(value):
        """
        Replace all integer numbers with special '<integer>' token
        :param value: input code
        :return: code with replaced floats
        """
        return integer_re.sub(INTEGER_REPLACEMENT, value)

    @staticmethod
    def strip_escaped_special_characters(value):
        """
        Remove escaped special character from the code
        :param value: input code
        :return: code without escaped special characters
        """
        return escaped_special_characters.sub('', value)

    @staticmethod
    def strip_multiple_spaces(value):
        """
        Strip all multiple spaces from the code
        :param value: input code
        :return: code without redundant spaces
        """
        return multiple_spaces_re.sub(' ', value)

    @staticmethod
    def strip_new_lines(value):
        """
        Strip all '\n' characters from the code
        :param value: input code
        :return: code without '\n' characters
        """
        return new_lines_re.sub(' ', value)

    @staticmethod
    def tokenize(value):
        """
        Tokenize the input code
        :param value: input code
        :return: code prepared for tokenization on ' '
        """
        res = re.split(r"([^a-zA-Z_])", value)
        res = ' '.join(res)
        return res
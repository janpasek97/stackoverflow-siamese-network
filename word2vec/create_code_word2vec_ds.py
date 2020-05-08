"""
Create word2vec corpus for code
"""

from elasticsearch_dsl import connections
from data.documents import ES_HOSTS, Post, ES_LOGIN
from network.utils.code_cleaner import CodeCleaner

import time


def create_code_word2vec_ds():
    """
    Prepares a code corpus for training a Word2Vec embeddings. The resulted corpus is made up of
    all code snippets from all the Stackoverflow posts. The output corpus is already pre-processed and no
    additional pre-processing is necessary.

    :return: void
    """
    connections.create_connection(hosts=ES_HOSTS, timeout=9999, http_auth=ES_LOGIN)
    time_start = time.time()

    code_cleaner = CodeCleaner()

    print("Creating dataset for code word2vec")

    # select all posts that comes from the Stackoverflow
    post_search = Post.search().filter("term", page="stackoverflow").params(scroll="1440m")
    posts = post_search.scan()

    with open("word2vec_code_ds.txt", "w", encoding="utf-8") as f:
        # export all pre-processed code snippets
        for i, post in enumerate(posts):
            try:
                if isinstance(post.text, str):
                    cleaned = code_cleaner.clean_code(post.text)
                    if cleaned != '':
                        f.write(cleaned)
                        f.write("\n")
            except UnicodeEncodeError:
                continue
            if i % 10000 == 0 and i != 0:
                print(f"\tProcessing {i}. post.")

    time_end = time.time()
    time_total = time_end - time_start
    print("Dataset created successfully ...")
    print(f"Dataset creation process took {int(time_total / 60)} min and {int(time_total % 60)} seconds")


if __name__ == '__main__':
    create_code_word2vec_ds()

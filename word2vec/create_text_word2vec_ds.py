"""
Create word2vec corpus for text
"""
from elasticsearch_dsl import connections
from data.documents import ES_HOSTS, Post, ES_LOGIN
from network.utils.text_cleaner import TextCleaner

import time


def create_text_word2vec_ds():
    """
    Prepares a text corpus for training a Word2Vec embeddings. The resulted corpus is made up of
    all the Stackoverflow posts. The output corpus is already pre-processed and no
    additional pre-processing is necessary.

    :return: void
    """
    connections.create_connection(hosts=ES_HOSTS, timeout=9999, http_auth=ES_LOGIN)
    time_start = time.time()

    tcleaner = TextCleaner()

    print("Creating dataset for text word2vec")

    # select all posts that comes from the Stackoverflow
    post_search = Post.search().filter("term", page="stackoverflow").params(scroll="1440m")
    posts = post_search.scan()

    with open("word2vec_text_ds.txt", "w", encoding="utf-8") as f:
        # export all pre-processed posts
        for i, post in enumerate(posts):
            try:
                if isinstance(post.text, str):
                    f.write(tcleaner.clean_text(post.text))
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
    create_text_word2vec_ds()

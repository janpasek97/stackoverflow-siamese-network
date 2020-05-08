"""
Functions for cleaning the dataset
"""
from data.documents import PostLink, Post
from elasticsearch import Elasticsearch
from elasticsearch_dsl import UpdateByQuery
from data.documents import ES_HOSTS, ES_LOGIN
import time


def _post_links_cleanup():
    """
    Searches for links between Posts that does not exists anymore and delete them from index

    :return: void
    """
    link_search = PostLink.search().filter("match", link_type=3).params(scroll="180m")
    links = link_search.scan()
    for i, link in enumerate(links):
        if Post.get_post(link.post_ID, link.page) is None or Post.get_post(link.related_post_ID, link.page) is None:
            link.delete()


def reset_dataset_flags(reset_types):
    """
    Resets all dataset assignments in Post index

    :return: void
    """
    print("Resetting dataset assignments ...")
    time_start_partial = time.time()
    es = Elasticsearch(ES_HOSTS, timeout=999999, http_auth=ES_LOGIN)
    update = UpdateByQuery(index="posts").using(es).filter("term", post_type=1).filter("terms", ds_item_role=reset_types).filter("term", is_in_ds=True).script(source="ctx._source.is_in_ds=false", lang="painless").params(conflicts="proceed")
    update.execute()
    time_partial = time.time() - time_start_partial
    print(f"    {int(time_partial/60)} min, {int(time_partial%60)} s")


def dataset_source_cleanup():
    """
    Perform all dataset cleanup operations

    :return: void
    """
    print("Cleaning up the dataset sources ...")
    time_start_partial = time.time()
    _post_links_cleanup()
    time_partial = time.time() - time_start_partial
    print(f"    {int(time_partial/60)} min, {int(time_partial%60)} s")

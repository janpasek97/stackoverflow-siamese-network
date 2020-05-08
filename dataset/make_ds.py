"""
Dataset creation procedures
"""
from data.documents import PostLink, Post, ES_HOSTS, ES_LOGIN
from elasticsearch_dsl.connections import connections
from elasticsearch.exceptions import RequestError
from dataset.dataset_cleanup import dataset_source_cleanup, reset_dataset_flags
from dataset.shuffle_and_split import shuffle_and_split
from multiprocessing import Pool
import time
import csv
import argparse


# MODULE CONSTANTS

DS_EXPORT_FILE = "ds_export.csv"        # output file

MAX_DUPLICATE_RECURSION_DEPTH = 5       # how deep transitive duplicate relationships to look for
POST_PER_SIMILAR_SEARCH = 3             # how many similar questions to search for for each master post
POSTS_PER_DIFFERENT_SEARCH = 3          # how many different questions to search for for each master post

# definition of dataset item role constants
DS_MAIN_POST = 0
DS_DUPLICATE = 1
DS_SIMILAR = 2
DS_DIFFERENT = 3

# parametrization into how many parallel threads to use for processing the data slices and into how many restartable
# parts to split the whole task
PARALLEL_SLICES = 10
JOB_PARTS = 12

# global current dataset size counter
ds_size = 1


def export_dataset_to_csv():
    """
    Exports the whole dataset into CSV in following format: "<first post ID>, <second post ID>, <label>".
    During the export, different posts are assigned to dataset.

    :return: N/A
    """
    with open(DS_EXPORT_FILE, "w", encoding="utf-8", newline="") as file:
        csv_data = [["first_post", "second_post", "label"]]
        writer = csv.writer(file)

        main_posts_search = Post.search().filter("term", is_in_ds=True).filter("term", ds_item_role=0).params(scroll="1440m")
        main_posts = main_posts_search.scan()

        # iterate over all main posts in dataset
        for post in main_posts:
            # export all duplicates and similar posts
            similar_posts_search = Post.search().filter("term", is_in_ds=True).filter("term", ds_group=post.ds_group).exclude("term", ds_item_role=0)
            similar_posts = similar_posts_search.scan()

            for p in similar_posts:
                csv_data.append([f"{post.post_ID}-{post.page}", f"{p.post_ID}-{p.page}", p.ds_item_role - 1])

            # find and export all different posts
            different_posts_search = Post.search().filter("term", is_in_ds=False)[0:POSTS_PER_DIFFERENT_SEARCH]
            different_posts_result = different_posts_search.execute()

            for p in different_posts_result.hits:
                csv_data.append([f"{post.post_ID}-{post.page}", f"{p.post_ID}-{p.page}", 2])

            writer.writerows(csv_data)
            csv_data = []


def add_similar_posts(post):
    """
    Search for similar posts for given post and add assigns them to dataset.

    :param post: post for which, similar posts will be processed
    :return: N/A
    """
    t = 0
    # infinite while loop provides a possibility to retry search for more times in case of error
    while True:
        t += 1
        try:
            similar_posts_search = Post.search().filter("term", post_type=1).filter("term", page=post.page).exclude("term", is_in_ds=True).query(
                "match", text=post.text)[0:POST_PER_SIMILAR_SEARCH]
            similar_posts_result = similar_posts_search.execute()
            for similar_post in similar_posts_result.hits:
                add_post_into_ds(similar_post, post.ds_group, DS_SIMILAR)
            break
        except RequestError:
            time.sleep(2)
            pass
        if t > 5:
            print("\n\nERROR")
            break


def add_duplicates_into_ds(duplicates, group_nr):
    """
    Add all given duplicate posts to the dataset into given DS group number

    :param duplicates: posts to be assigned as duplicates
    :param group_nr: dataset group number into which, posts will be assigned
    :return: N/A
    """
    for duplicate in duplicates:
        add_post_into_ds(duplicate, group_nr, DS_DUPLICATE)


def find_all_duplicates_for_post(post, recursion_depth=0):
    """
    Searches for all duplicates of given post in the whole dataset. Since the function searches for transitive relations
    as well, it is necessary to define the maximum recursion depth (MAX_DUPLICATE_RECURSION_DEPTH) where duplicates are
    going to be searched for.

    :param post: post for which duplicates shall be searched for
    :param recursion_depth: current recursion depth indication
    :return: List of all found duplicates
    """
    if recursion_depth > MAX_DUPLICATE_RECURSION_DEPTH:
        return []

    duplicates_result = []

    duplicates_search = PostLink.search().filter("term", link_type=3).filter("term", post_ID=post.post_ID).filter("term", page=post.page)
    duplicate_links = duplicates_search.scan()

    for duplicate_link in duplicate_links:
        duplicate_post = Post.get_post(duplicate_link.related_post_ID, duplicate_link.page)
        if not duplicate_post.is_in_ds:
            duplicates_result.append(duplicate_post)

            duplicates_of_duplicate = find_all_duplicates_for_post(duplicate_post, recursion_depth=recursion_depth+1)
            if len(duplicates_of_duplicate) != 0:
                duplicates_result.extend(duplicates_of_duplicate)

    return duplicates_result


def add_main_post_into_ds(post):
    """
    Add the given post and it's duplicates into dataset

    :param post: main posts which will be added into dataset with it's duplicates
    :return: N/A
    """
    global ds_size
    if post is not None and not post.is_in_ds:
        duplicates = find_all_duplicates_for_post(post)
        if (len(duplicates)) > 0:
            add_post_into_ds(post, ds_size, DS_MAIN_POST)
            add_duplicates_into_ds(duplicates, ds_size)
            ds_size += 1


def add_post_into_ds(post, ds_group, ds_item_role):
    """
    Add given post into dataset and assigns it to given dataset group with given ds item role (main/duplicate/similar/different).

    :param post: posts to be add into dataset
    :param ds_group: dataset group into which the post will be assigned
    :param ds_item_role: post role in dataset group
    :return: N/A
    """
    post.ds_group = ds_group
    post.ds_item_role = ds_item_role
    post.is_in_ds = True
    post.save()


def process_similar_posts(slice_idx):
    """
    Process given slice of similar posts in the dataset - search for similarities and assign the into ds.

    :param slice_idx: number of slice that will be processed
    :return: N/A
    """
    global part
    main_posts_in_ds_search = Post.search().filter("term", post_type=1).filter("term", ds_item_role=0).filter("term", is_in_ds=True).params(scroll="1440m")
    main_posts_in_ds_search = main_posts_in_ds_search.extra(slice={"id": part * PARALLEL_SLICES + slice_idx, "max": PARALLEL_SLICES*JOB_PARTS})
    posts = main_posts_in_ds_search.scan()

    for i, main_post in enumerate(posts):
        add_similar_posts(main_post)


def make_dataset(arguments):
    """
    Perform steps of dataset creation processed according to given commandline arguments.

    :param arguments: parsed command line arguments
    :return: N/A
    """
    connections.create_connection(hosts=ES_HOSTS, timeout=9999, http_auth=ES_LOGIN)
    time_start = time.time()

    # cleanup of the invalid dataset duplicate links
    if arguments.clean:
        dataset_source_cleanup()

    # reset dataset assignments of posts with given roles
    if arguments.reset is not None and len(arguments.reset) > 0:
        reset_dataset_flags(arguments.reset)

    # create dataset base from main posts and their duplicates
    if arguments.base:
        link_search = PostLink.search().filter("term", link_type=3).params(scroll="1440m")
        links = link_search.scan()

        time_start_partial = time.time()
        print("Creating dataset base from duplicates ...")
        for i, link in enumerate(links):
            add_main_post_into_ds(Post.get_post(link.post_ID, link.page))
            if i % 10 == 0:
                print(f"\r    Processing - {i}", end="")
        time_partial = time.time() - time_start_partial
        print(f"    {int(time_partial / 60)} min, {int(time_partial % 60)} s")
        print()

    # search and assign similar posts to all main posts in dataset
    if arguments.similar is not None:
        time_start_partial = time.time()
        print("Getting similar posts for posts in dataset base ...")
        print(f"     Part: {arguments.similar}")
        pool = Pool(PARALLEL_SLICES)
        pool.map(process_similar_posts, range(PARALLEL_SLICES))

        time_partial = time.time() - time_start_partial
        print(f"    {int(time_partial / 60)} min, {int(time_partial % 60)} s")
        print()

    # export the dataset to CSV file
    if arguments.export:
        time_start_partial = time.time()
        print("Exporting whole dataset to general csv...")
        export_dataset_to_csv()
        time_partial = time.time() - time_start_partial
        print(f"    {int(time_partial / 60)} min, {int(time_partial % 60)} s")

        time_start_partial = time.time()
        print("Shuffling and splitting the general csv into train, dev and test parts...")
        shuffle_and_split(DS_EXPORT_FILE)
        time_partial = time.time() - time_start_partial
        print(f"    {int(time_partial / 60)} min, {int(time_partial % 60)} s")

    time_end = time.time()
    time_total = time_end - time_start
    print("Dataset created successfully ...")
    print(f"Dataset creation process took {int(time_total / 60)} min and {int(time_total % 60)} seconds")


if __name__ == '__main__':
    """
    Parses command line arguments and call make_dataset method which decides which steps to be done
    """
    parser = argparse.ArgumentParser("Creates complete dataset from all relevant StackOverflow posts")
    parser.add_argument("-c", "--clean", action="store_true", help="Clean invalid links from dataset")
    parser.add_argument("-r", "--reset", action="append", help="Dataset item types which's assignments shall be reset", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("-b", "--base", help="Create dataset base from duplicate", action="store_true")
    parser.add_argument("-s", "--similar", help="Add given part of similar posts to dataset", type=int, choices=[i for i in range(12)])
    parser.add_argument("-e", "--export", help="Export dataset to csv files", action="store_true")

    args = parser.parse_args()

    if args.similar is not None:
        part = args.similar

    make_dataset(args)

from data.documents import Post


def __search_fulltext(search_text, post_start, post_end, request, date_filter):
    """
    Perform fulltext search in order to find posts that match with given search_text string

    :param search_text: text from user for which all matched are gonna be searched for
    :param post_start: post search start offset - pagination parameter
    :param post_end: post search end offset - pagination parameter
    :param request: request object which carries important GET and POST parameters
    :param date_filter: date filter settings for search
    :return: display information of all found articles
    """
    posts_search = Post.search().filter("term", post_type=1).query("multi_match", query=search_text,
                                                                   fields=["title", "text"])[post_start:post_end]
    # determine filters
    if request.GET.getlist("pages") and request.GET.get("pages", "all") != "all": # pages filter
        pages = request.GET.getlist("pages")
        posts_search = posts_search.filter("terms", page=pages)

    if request.GET.get("with_answer", None) is not None:  # with answer only filter
        posts_search = posts_search.filter("exists", field="accepted_answer_ID")

    # date filter
    posts_search = posts_search.filter("range", creation_date={"gte": date_filter["start"], "lt": date_filter["end"]})  # date filter always has at least a default value

    posts_response = posts_search.execute()
    return Post.get_display_info_for_posts(posts_response.hits)


def __search_siamese(search_text, post_start, post_end):
    """
    Siamese search is not implemented yet!!! Only placeholder function....

    :param search_text:
    :param post_start:
    :param post_end:
    :return:
    """
    return []


def search(search_type, search_text, page, posts_per_page, request, date_filter):
    """
    Perform selected type of post search and return display information of all resulted posts

    :param search_type: type of search that is gonna be performed - values: {fulltext, siamese}
    :param search_text: text from user for which all matched are gonna be searched for
    :param page: page from which the articles are gonna be searched for
    :param posts_per_page: how many posts shall be found
    :param request: request object which carries important GET and POST parameters
    :param date_filter: date filter settings for search
    :return: display information of all found articles
    """
    if search_text == "":
        return []
    post_start = (page - 1) * posts_per_page
    post_end = (page * posts_per_page) + 1
    if search_type == "fulltext":
        return __search_fulltext(search_text, post_start, post_end, request, date_filter)
    else:
        return __search_siamese(search_text, post_start, post_end)
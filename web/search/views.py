from django.shortcuts import render
from django.http import Http404, HttpResponse
from django.template.loader import render_to_string
from elasticsearch_dsl.connections import connections

from data.documents import Post, User, Comment, ES_HOSTS, ES_LOGIN
from data.utils import sanitize_html_for_web
from .search import search

import datetime
import json

POST_PER_PAGE = 15


# Create your views here.
def index(request):
    """
    Index view controller

    :param request: http request object
    :return: rendered page
    """
    return render(request, "search/index.html", None)


def question_search_pagination(request, page):
    """
    Question search with pagination view controller

    :param request: http request object
    :param page: currently selected page
    :return: rendered page
    """
    form = {}
    url_params = request.GET.copy().urlencode()

    # get GET parameters
    search_type = request.GET.get("search_type", None)
    search_text = request.GET.get("search_text", None)

    # define the required search type
    if search_type is None:
        if "search_type" in request.COOKIES:
            search_type = request.COOKIES["search_type"]
        else:
            search_type = "fulltext"

    if search_text is None:
        search_text = ""

    page = page if page > 0 else 1

    # determine filters
    if request.GET.getlist("pages") and request.GET.get("pages", "all") != "all":
        pages = request.GET.getlist("pages")
        form["pages"] = pages

    if request.GET.get("with_answer", None) is not None:
        form["with_answer"] = True

    if request.GET.get("date_range_start", None) and request.GET.get("date_range_end", None):
        date_start_string = request.GET["date_range_start"]
        date_end_string = request.GET["date_range_end"]
        date_start = datetime.datetime.strptime(date_start_string, "%Y-%m-%d")
        date_end = datetime.datetime.strptime(date_end_string, "%Y-%m-%d")
    else:
        date_start = datetime.datetime.now() - datetime.timedelta(days=730)
        date_end = datetime.datetime.now()

    form["date_range_start"] = str(date_start.date())
    form["date_range_end"] = str(date_end.date())

    # pagination settings and feed dict
    pagination_info = {"page_nr": page,
                       "next_page_nr": page + 1,
                       "previous_page_nr": page - 1,
                       "url_params": url_params}

    # render the template
    response = render(request, "search/question_search.html",
                      context={"pagination_info": pagination_info, "display_search": True,
                               "search_text": search_text, "form": form})

    response.set_cookie("search_type", search_type)
    return response


def question_search(request):
    """
    Question search without pagination view controller

    :param request: http request object
    :return: rendered page
    """
    return question_search_pagination(request, 1)


def explore_questions_pagination(request, page):
    """
    Explore question with pagination view controller

    :param request: http request
    :param page: currently selected page
    :return: rendered page
    """
    form = {}
    url_params = request.GET.copy().urlencode()

    # determine which posts should be shown according to selected page
    page = page if page > 0 else 1
    post_start = (page - 1) * POST_PER_PAGE
    post_end = (page * POST_PER_PAGE) + 1

    # determine filters
    if request.GET.getlist("pages") and request.GET.get("pages", "all") != "all":
        pages = request.GET.getlist("pages")
        form["pages"] = pages

    if request.GET.get("with_answer", None) is not None:
        form["with_answer"] = True

    if request.GET.get("date_range_start", None) and request.GET.get("date_range_end", None):
        date_start_string = request.GET["date_range_start"]
        date_end_string = request.GET["date_range_end"]
        date_start = datetime.datetime.strptime(date_start_string, "%Y-%m-%d")
        date_end = datetime.datetime.strptime(date_end_string, "%Y-%m-%d")
    else:
        date_start = datetime.datetime.now() - datetime.timedelta(days=730)
        date_end = datetime.datetime.now()

    form["date_range_start"] = str(date_start.date())
    form["date_range_end"] = str(date_end.date())

    # pagination settings and feed dict
    has_previous = True if page > 1 else False
    has_next = True
    pagination_info = {"has_previous": has_previous, "has_next": has_next, "page_nr": page, "next_page_nr": page + 1,
                       "previous_page_nr": page - 1, "url_params": url_params}

    # render the template
    return render(request, "search/explore_questions.html",
                  context={"pagination_info": pagination_info, "display_search": True, "form": form})


def explore_questions(request):
    """
    Explore question without pagination view controller

    :param request: http request object
    :return: rendered page
    """
    return explore_questions_pagination(request, 1)


def detail(request, post_id, page):
    """
    Post detail view controller

    :param request: http request object
    :param post_id: id for which the detail is displayed
    :param page: page of article
    :return: rendered page
    """
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)
    post_result = Post.get_post(post_id, page)

    if post_result is None:
        raise Http404

    # get accepted answer for given question and create feed dict for that
    accepted_answer = None
    if post_result.accepted_answer_ID is not None:
        accepted_answer_result = Post.get_post(post_result.accepted_answer_ID, page)
        accepted_answer = {"author": User.get_user_display_name(accepted_answer_result.owner_ID, page),
                           "date": accepted_answer_result.creation_date,
                           "text": sanitize_html_for_web(accepted_answer_result.text),
                           "post_ID": accepted_answer_result.post_ID}

    # create feed dict for template
    post = {"title": post_result.title, "date": post_result.creation_date,
            "text": sanitize_html_for_web(post_result.text),
            "author": User.get_user_display_name(post_result.owner_ID, page)}

    # search for answers and comments for given question
    except_id = accepted_answer['post_ID'] if accepted_answer is not None else None
    answers = Post.get_post_answers_information(post_result.post_ID, page, except_id)
    comments = Comment.get_post_comments_information(post_result.post_ID, page)

    # render the template
    return render(request, 'search/question_details.html', context={"display_search": True,
                                                                    "post": post,
                                                                    "accepted_answer": accepted_answer,
                                                                    "answers": answers,
                                                                    "comments": comments})


def question_search_content_loader(request, page=1):
    """
    Searches for the the posts based on given queries and return selected page of results as a
    JSON objects. It is used by AJAX

    :param request: http request object
    :param page: page of information for pagination
    :return: data dictionary as a JSON object that is the processed by AJAX
    """
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)

    # get GET parameters
    search_type = request.GET.get("search_type", None)
    search_text = request.GET.get("search_text", None)

    # define the required search type
    if search_type is None:
        if "search_type" in request.COOKIES:
            search_type = request.COOKIES["search_type"]
        else:
            search_type = "fulltext"

    if search_text is None:
        search_text = ""

    page = page if page > 0 else 1

    # determine filters
    if request.GET.get("date_range_start", None) and request.GET.get("date_range_end", None):
        date_start_string = request.GET["date_range_start"]
        date_end_string = request.GET["date_range_end"]
        date_start = datetime.datetime.strptime(date_start_string, "%Y-%m-%d")
        date_end = datetime.datetime.strptime(date_end_string, "%Y-%m-%d")
    else:
        date_start = datetime.datetime.now() - datetime.timedelta(days=730)
        date_end = datetime.datetime.now()

    date_filter = {"start": date_start, "end": date_end}

    # perform search
    posts = search(search_type, search_text, page, POST_PER_PAGE, request, date_filter)

    # pagination settings and feed dict
    has_previous = True if page > 1 else False
    has_next = True if len(posts) > POST_PER_PAGE else False

    # render the template
    html = render_to_string('search/posts_displays.html', {"posts": posts[:-1]})
    json_dict = {"html": html, "has_next_page": has_next, "has_previous_page": has_previous}

    response = HttpResponse(json.dumps(json_dict))
    response.set_cookie("search_type", search_type)
    return response


def explore_questions_content_loader(request, page=1):
    """
    Return selected page of results (from the newest) as a JSON objects. It is used by AJAX.

    :param request: http request object
    :param page: page of information for pagination
    :return: data dictionary as a JSON object that is the processed by AJAX
    """
    # determine which posts should be shown according to selected page
    page = page if page > 0 else 1
    post_start = (page - 1) * POST_PER_PAGE
    post_end = (page * POST_PER_PAGE) + 1

    has_previous_page = True if page > 1 else False

    # search for posts that will be show
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)
    s = Post.search().filter("term", post_type=1).sort({"creation_date": {"order": "desc"}})[post_start:post_end]

    # determine filters
    if request.GET.getlist("pages[]") and request.GET.get("pages[]", "all") != "all":
        pages = request.GET.getlist("pages[]")
        s = s.filter("terms", page=pages)

    if request.GET.get("with_answer", None) is not None:
        s = s.filter("exists", field="accepted_answer_ID")

    if request.GET.get("date_range_start", None) and request.GET.get("date_range_end", None):
        date_start_string = request.GET["date_range_start"]
        date_end_string = request.GET["date_range_end"]
        date_start = datetime.datetime.strptime(date_start_string, "%Y-%m-%d")
        date_end = datetime.datetime.strptime(date_end_string, "%Y-%m-%d")
    else:
        date_start = datetime.datetime.now() - datetime.timedelta(days=730)
        date_end = datetime.datetime.now()

    s = s.filter("range", creation_date={"gte": date_start, "lt": date_end})

    response = s.execute()

    # get information about posts
    posts = Post.get_display_info_for_posts(response.hits)

    has_next_page = True if len(posts) > POST_PER_PAGE else False

    html = render_to_string('search/posts_displays.html', {"posts": posts[:-1]})
    json_dict = {"html": html, "has_next_page": has_next_page, "has_previous_page": has_previous_page}
    return HttpResponse(json.dumps(json_dict))

"""
Elasticsearch index definitions for elasticsearch-dsl
"""
from elasticsearch_dsl import Document, Date, Keyword, Text, Long, Boolean, Integer, analyzer
from elasticsearch_dsl.connections import connections
from data.utils import sanitize_html_for_web

ES_HOSTS = ["xxx"]
ES_LOGIN = "xxx"

html_strip = analyzer("html_strip", tokenizer="standard", filter=["stop", "lowercase", "snowball"],
                      char_filter=["html_strip"])


class Comment(Document):
    """
    Comment index structure and operation specification
    """
    creation_date = Date()
    comment_ID = Long()
    page = Text(analyzer=html_strip, fields={"raw": Keyword()})
    post_ID = Long()
    text = Text(analyzer=html_strip)
    user_ID = Long()

    class Index:
        name = "comments"

    @staticmethod
    def get_post_comments_information(post_id, page):
        """
        Query all information about comments for post with given ID from given page

        :param post_id: post ID for which all comments are gonna be found
        :param page: page on which the post was published
        :return: list of dictionaries with information about found comments
        """
        comments_search = Comment.search().filter("term", post_ID=post_id).filter("match", page=page).sort(
            {"creation_date": {"order": "desc"}})
        comments_results = comments_search.execute()
        comments = []

        for comment in comments_results.hits:
            author = User.get_user_display_name(comment.user_ID, page)
            comments.append(
                {"author": author, "date": comment.creation_date, "text": sanitize_html_for_web(comment.text)})

        return comments


class Post(Document):
    """
    Post index structure and operation specification
    """
    post_ID = Long()
    post_type = Long()
    parrent_ID = Long()
    accepted_answer_ID = Long()
    creation_date = Date()
    text = Text(analyzer=html_strip)
    owner_ID = Long()
    title = Text(analyzer=html_strip)
    page = Text(analyzer="standard", fields={"raw": Keyword()})
    is_in_ds = Boolean()
    ds_group = Long()
    ds_item_role = Integer()

    class Index:
        name = "posts"

    @staticmethod
    def get_post(post_id, page):
        """
        Query post with given ID from given page

        :param post_id: post ID which will be searched for
        :param page: page where the post was published
        :return: found post object
        """
        post_search = Post.search().filter("term", post_ID=post_id).filter("match", page=page)
        post_results = post_search.execute()
        try:
            return post_results.hits[0]
        except IndexError:
            return None

    @staticmethod
    def get_display_info_for_post(post):
        """
        Get all information about given post
            - title
            - text
            - author name
            - number of comments
            - number of answers
            - page
            - post ID
            - accepted answer ID

        :param post: post for which information are gonna be found
        :return: dictionary with all relevant information
        """
        text = sanitize_html_for_web(post.text, display_code=False)

        # search for author
        if post.owner_ID is None:
            username = "unknown"
        else:
            username = User.get_user_display_name(post.owner_ID, post.page)

        # get comment count
        comments = Comment.get_post_comments_information(post.post_ID, post.page)
        comment_count = len(comments)

        # get answer count
        answers = Post.get_post_answers_information(post.post_ID, post.page)
        answer_count = len(answers)

        return {"comment_count": comment_count + answer_count, "author_name": username, "page": post.page,
                "post_ID": post.post_ID, "title": post.title, "accepted_answer_ID": post.accepted_answer_ID,
                "text": text}

    @staticmethod
    def get_display_info_for_posts(posts):
        """
        Get all information about given posts
            - title
            - text
            - author name
            - number of comments
            - number of answers
            - page
            - post ID
            - accepted answer ID

        :param posts: list of all posts for which information are gonna be found
        :return: dictionary with all relevant information
        """
        posts_info = []
        for post in posts:
            posts_info.append(Post.get_display_info_for_post(post))
        return posts_info

    @staticmethod
    def get_post_answers_information(post_id, page, except_id=None):
        """
        Query information about all answers for given post from given page

        :param post_id: post ID which's answers are gonna be found
        :param page: page where answers shall be searched for
        :param except_id: answer IDs which shall be excluded from search
        :return: list of dictionaries with information about found answers
        """
        answers_search = Post.search().filter("term", parrent_ID=post_id).filter("match", page=page).sort(
            {"creation_date": {"order": "desc"}})
        answers_results = answers_search.execute()
        answers = []

        for answer in answers_results.hits:
            if except_id is not None and answer.post_ID == except_id:
                continue
            author = User.get_user_display_name(answer.owner_ID, page)
            answers.append({"author": author, "date": answer.creation_date, "text": sanitize_html_for_web(answer.text)})

        return answers

    @staticmethod
    def is_post_used(post_id, page):
        """
        Get information if post is already used in dataset

        :param post_id: ID of post for which usage is going to be found
        :param page: post's page
        :return: True if post is already used in DS, False if post is not used or does not exists
        """
        post_search = Post.search().filter("term", post_ID=post_id).filter("term", page=page)
        post_response = post_search.execute()
        if len(post_response.hits) > 0:
            selected_post = post_response.hits[0]
            return selected_post.is_in_ds

        return False


class PostLink(Document):
    """
    PostLink index structure and operation specification
    """
    link_ID = Long()
    creation_date = Date()
    post_ID = Long()
    related_post_ID = Long()
    link_type = Long()
    page = Text(analyzer="standard", fields={"raw": Keyword()})

    class Index:
        name = "links"


class User(Document):
    """
    User index structure and operation specification
    """
    user_ID = Long()
    display_name = Text()
    creation_date = Date()
    page = Text(analyzer="standard", fields={"raw": Keyword()})

    class Index:
        name = "users"

    @staticmethod
    def get_user_display_name(user_id, page):
        """
        Query display name of user with given ID from given page

        :param user_id: ID of user for whose display name are gonna be found
        :param page: user's page
        :return: user's display name as a string
        """
        if user_id is None:
            return "unknown"

        user_search = User.search().filter("term", user_ID=user_id).filter("match", page=page)
        user_results = user_search.execute()
        try:
            user_result = user_results.hits[0]
            return user_result.display_name
        except IndexError:
            return "unknown"


if __name__ == '__main__':
    # Initialization of specified indexes
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)

    Comment.init()
    Post.init()
    PostLink.init()
    User.init()

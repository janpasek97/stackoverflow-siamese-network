"""
Data utilities
"""
from bs4 import BeautifulSoup

VALID_TAGS = ["strong", "em", "p", "ul", "li", "br", "code", "pre"]


def sanitize_html_for_web(value, display_code=True):
    """
    Sanitize HTML from all unwanted HTML tags. All <pre><code>...</code></pre>
    sections are replaced with placeholder text is display_code is set to False

    :param value: string with text that needs to be cleaned
    :param display_code: if is True, then the <pre><code>...</code></pre> will not be removed from resulted text
    :return: sanitized text
    """
    soup = BeautifulSoup(value, "html5lib")

    for tag in soup.find_all(True):
        if tag.name not in VALID_TAGS:
            tag.hidden = True
        elif tag.name == "code" and not display_code and tag.parent is not None:
            if tag.parent.name == "pre":
                tag.string.replace_with("Inserted code --- see details for code expansion")

    return str(soup)











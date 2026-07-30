from urllib.parse import urlencode

from django import template


register = template.Library()


@register.simple_tag(takes_context=True)
def pagination_url(context, page_param, page_number, anchor_id=""):
    """Replace one page parameter while retaining all other query parameters."""
    request = context["request"]
    params = request.GET.copy()
    params[page_param] = page_number
    query_string = urlencode(list(params.lists()), doseq=True)
    anchor = f"#{anchor_id}" if anchor_id else ""
    return f"{request.path}?{query_string}{anchor}"

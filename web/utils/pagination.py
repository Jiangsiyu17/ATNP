# web/utils/pagination.py
from django.core.paginator import Paginator

def paginate_list(request, data_list, per_page=15):
    """
    对 Python list 进行分页（适用于 aggregate 后结果）
    """
    paginator = Paginator(data_list, per_page)
    page_number = (
        request.GET.get("page")
        or request.POST.get("page")
        or 1
    )
    page_obj = paginator.get_page(page_number)
    return page_obj

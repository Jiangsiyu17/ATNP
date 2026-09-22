import math

from django.shortcuts import render

from .services.literature import LiteratureDatabaseError
from .services.knowledge import (
    PAGE_SIZE as KNOWLEDGE_PAGE_SIZE,
    get_relations,
    search_knowledge,
)


def knowledge_search(request):
    head = request.GET.get("head", "").strip()
    relation = request.GET.get("relation", "").strip()
    tail = request.GET.get("tail", "").strip()
    searched = any((head, relation, tail))
    try:
        page = max(int(request.GET.get("page", 1)), 1)
    except (TypeError, ValueError):
        page = 1

    context = {
        "head": head, "relation": relation, "tail": tail,
        "searched": searched, "relations": [], "triples": [],
        "total": 0, "page": page, "total_pages": 0, "error": "",
    }
    try:
        context["relations"] = get_relations()
        if searched:
            results = search_knowledge(head, relation, tail, page=page)
            context.update(results)
            context["total_pages"] = math.ceil(results["total"] / KNOWLEDGE_PAGE_SIZE)
            if page > 1 and results["total"] and not results["triples"]:
                context["page"] = context["total_pages"]
                results = search_knowledge(
                    head, relation, tail, page=context["page"]
                )
                context.update(results)
    except LiteratureDatabaseError as exc:
        context["error"] = str(exc)

    return render(request, "web/knowledge_search.html", context)

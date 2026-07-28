## 化合物所在文献展示查询逻辑
import json
import re

from django.conf import settings

from .pubmed_loader import PUBMED_INDEX


FILE_MAP = {
    "N": settings.BASE_DIR / "crawler" / "5_N_all_clean_pubmed_results.jsonl",
    "P": settings.BASE_DIR / "crawler" / "5_P_all_clean_pubmed_results.jsonl",
}

def normalize_title(title):
    """标题标准化（用于去重）"""
    if not title:
        return ""
    return " ".join(title.lower().strip().split())


def normalize_name(name):
    if not name:
        return ""

    name = name.lower().strip()
    name = re.sub(r"^\(.*?\)-", "", name)
    name = " ".join(name.split())

    return name

def get_pubmed_papers(compound_name):
    name = normalize_name(compound_name)

    if name not in PUBMED_INDEX:
        return []

    papers = []

    seen_pmids = set()
    seen_titles = set()   # ✅ 新增：标题去重

    for item in PUBMED_INDEX[name]:
        file_path = FILE_MAP.get(item.get("file"))
        if file_path is None or not file_path.is_file():
            continue

        with file_path.open("r", encoding="utf-8") as f:
            f.seek(item["offset"])
            line = f.readline()

            try:
                data = json.loads(line)

                for p in data.get("papers", []):
                    pmid = p.get("pmid")
                    title = p.get("title", "")

                    if not pmid:
                        continue

                    norm_title = normalize_title(title)

                    # ✅ 双重去重
                    if pmid in seen_pmids:
                        continue
                    if norm_title in seen_titles:
                        continue

                    seen_pmids.add(pmid)
                    seen_titles.add(norm_title)

                    papers.append(p)

            except:
                continue

    return papers

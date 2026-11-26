from app.performance import timed
import html
import re
import unicodedata
from difflib import SequenceMatcher


def _normalize_doi(value):
    doi = str(value or "").strip().lower()
    doi = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        doi
    )
    return doi


def _reference_key(evidence):
    """优先通过 DOI、PMID 和 paper_id 判断是否为同一篇论文。"""

    doi = _normalize_doi(evidence.get("doi"))
    pmid = str(evidence.get("pmid") or "").strip()
    paper_id = str(evidence.get("paper_id") or "").strip()
    title = str(evidence.get("title") or "").strip().casefold()
    evidence_id = str(
        evidence.get("evidence_id") or ""
    ).strip()
    evidence_sentence = str(
        evidence.get("evidence_sentence") or ""
    ).strip()
    edge_id = str(evidence.get("edge_id") or "").strip()

    if doi:
        return ("doi", doi)
    if pmid:
        return ("pmid", pmid)
    if paper_id:
        return ("paper_id", paper_id)
    if title:
        return ("title", title)

    if evidence_id:
        return ("evidence_id", evidence_id)

    # 元数据全空时不应把所有 evidence 误合并成一篇论文。
    return ("evidence_text", edge_id, evidence_sentence)


def _normalize_evidence_sentence(value):
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _near_duplicate_sentence(left, right):
    left = _normalize_evidence_sentence(left)
    right = _normalize_evidence_sentence(right)

    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < 40:
        return False

    sequence_ratio = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    token_jaccard = (
        len(left_tokens & right_tokens) / len(union)
        if union else 0.0
    )
    return sequence_ratio >= 0.92 or token_jaccard >= 0.90


def _evidence_type_hints(reference):
    searchable = " ".join([
        str(reference.get("title") or ""),
        *[
            str(item.get("sentence") or "")
            for item in reference.get("evidence", [])
        ]
    ]).casefold()
    hints = []

    patterns = [
        ("review", ("review", "systematic review", "meta-analysis")),
        ("computational", ("docking", "in silico", "molecular dynamics")),
        ("formulation", ("nanoparticle", "nanoframework", "encapsulated", "formulation")),
        ("clinical", ("patient", "clinical trial", "randomized")),
        ("animal", ("mice", "mouse", "rats", "rat model", "xenograft")),
        ("cell", (" cells", "cell line", "in vitro")),
        ("plant isolation", ("isolated from", "extracted from", "plant"))
    ]

    for label, terms in patterns:
        if any(term in searchable for term in terms):
            hints.append(label)

    return hints or ["unspecified"]


def _relation_record(item, edge):
    record = {
        "edge_id": item.get("edge_id"),
        "relation": edge.get("relation"),
        "target": edge.get("tail_name"),
        "target_type": edge.get("tail_type")
    }

    if edge.get("plant_entity"):
        record["plant_entity"] = edge["plant_entity"]

    return record


def _evidence_provenance(item):
    return {key: item.get(key) for key in (
        'evidence_id', 'retrieval_source', 'source_passage_id', 'source_path',
        'source_archive', 'source_line', 'section_index', 'start_char', 'end_char', 'section')}


@timed("reference_building")
def build_references(evidence, edges):
    """
    按论文去重并分配 R1、R2... 结构化引用编号。

    同一篇论文可以包含多条证据句与多个 KG relation。
    """

    edge_map = {
        source_edge_id: edge
        for edge in edges
        for source_edge_id in (
            edge.get("source_edge_ids") or [edge.get("edge_id")]
        )
        if source_edge_id
    }

    reference_map = {}
    references = []

    for item in evidence:
        key = _reference_key(item)

        if key not in reference_map:
            doi = _normalize_doi(item.get("doi"))
            pmid = str(item.get("pmid") or "").strip()
            reference_id = f"R{len(references) + 1}"

            url = None
            if doi:
                url = f"https://doi.org/{doi}"
            elif pmid:
                url = (
                    "https://pubmed.ncbi.nlm.nih.gov/"
                    f"{pmid}/"
                )

            reference = {
                "id": reference_id,
                "title": item.get("title"),
                "doi": doi or None,
                "pmid": pmid or None,
                "paper_id": item.get("paper_id"),
                "pmcid": item.get("pmcid"),
                "url": url or item.get("source_url"),
                "intent_groups": [],
                "evidence": []
            }

            reference_map[key] = reference
            references.append(reference)

        reference = reference_map[key]
        item_intents = list(dict.fromkeys(
            item.get("intent_groups") or []
        ))
        reference["intent_groups"] = list(dict.fromkeys([
            *reference.get("intent_groups", []),
            *item_intents
        ]))
        edge = edge_map.get(item.get("edge_id"), {})
        sentence = html.unescape(str(item.get("evidence_sentence") or ""))
        relation_record = _relation_record(item, edge)
        duplicate = next((
            record
            for record in reference["evidence"]
            if _near_duplicate_sentence(record.get("sentence"), sentence)
        ), None)

        if duplicate is not None:
            provenance = _evidence_provenance(item)
            if provenance not in duplicate['provenance']:
                duplicate['provenance'].append(provenance)
            duplicate["evidence_ids"] = list(dict.fromkeys([
                *duplicate.get("evidence_ids", []),
                item.get("evidence_id")
            ]))
            duplicate["sections"] = list(dict.fromkeys([
                *duplicate.get("sections", []),
                item.get("section")
            ]))
            if relation_record not in duplicate["relations"]:
                duplicate["relations"].append(relation_record)
            duplicate["intent_groups"] = list(dict.fromkeys([
                *duplicate.get("intent_groups", []),
                *item_intents
            ]))
            continue

        reference["evidence"].append({
            "evidence_id": item.get("evidence_id"),
            "evidence_ids": [item.get("evidence_id")],
            "provenance": [_evidence_provenance(item)],
            "sentence": sentence,
            "retrieval_source": item.get("retrieval_source", "knowledge_graph"),
            "source_edge_id": item.get("source_edge_id", item.get("edge_id")),
            "source_url": item.get("source_url"),
            "source_passage_id": item.get("source_passage_id"),
            "source_path": item.get("source_path"),
            "source_archive": item.get("source_archive"),
            "source_line": item.get("source_line"),
            "section_index": item.get("section_index"),
            "start_char": item.get("start_char"),
            "end_char": item.get("end_char"),
            "text_source": item.get("text_source"),
            "page": item.get("page"),
            "section": item.get("section"),
            "sections": [item.get("section")],
            "confidence": item.get("confidence"),
            "edge_id": item.get("edge_id"),
            "relation": edge.get("relation"),
            "target": edge.get("tail_name"),
            "target_type": edge.get("tail_type"),
            "intent_groups": item_intents,
            "relations": [relation_record]
        })

    for reference in references:
        reference["evidence_type_hints"] = _evidence_type_hints(reference)

    return references

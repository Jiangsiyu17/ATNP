from app.performance import timed
import math
import re
import unicodedata
from collections import Counter, defaultdict

from app.evidence_grounding import (
    filter_evidence_by_compound_attribution
)


INTENT_EXPANSIONS = {
    "compound_identity": [
        "compound identity",
        "drug class",
        "prodrug",
        "defined as",
        "selective inhibitor"
    ],
    "anticancer_activity": [
        "anticancer",
        "antitumor",
        "cancer",
        "tumor",
        "carcinoma"
    ],
    "anticancer_mechanism": [
        "anticancer",
        "antitumor",
        "cancer",
        "tumor",
        "apoptosis",
        "proliferation",
        "metastasis",
        "angiogenesis",
        "cell cycle",
        "signaling pathway"
    ],
    "pharmacological_effect": [
        "pharmacological effect",
        "pharmacological activity",
        "therapeutic effect",
        "biological activity"
    ],
    "disease_association": [
        "disease association",
        "associated with",
        "disease",
        "disorder",
        "indication",
        "treatment",
        "therapeutic"
    ],
    "target_mechanism": [
        "molecular target",
        "protein target",
        "gene",
        "signaling pathway",
        "molecular mechanism",
        "binding"
    ],
    "plant_source": [
        "plant source",
        "isolated from",
        "extracted from",
        "derived from",
        "species"
    ],
    "toxicity": [
        "toxicity",
        "toxic effect",
        "adverse effect",
        "side effect",
        "safety"
    ],
    "pharmacokinetics": [
        "pharmacokinetics",
        "absorption",
        "distribution",
        "metabolism",
        "excretion",
        "bioavailability"
    ]
}


CHINESE_QUERY_EXPANSIONS = {
    "是什么": ["compound identity", "drug class", "prodrug", "defined as"],
    "什么是": ["compound identity", "drug class", "prodrug", "defined as"],
    "抗肿瘤": ["anticancer", "antitumor", "cancer", "tumor"],
    "抗癌": ["anticancer", "antitumor", "cancer"],
    "肿瘤": ["cancer", "tumor", "tumour"],
    "癌症": ["cancer", "carcinoma", "tumor"],
    "机制": ["mechanism", "pathway", "signaling"],
    "靶点": ["target", "protein", "gene"],
    "通路": ["pathway", "signaling pathway"],
    "药理": ["pharmacological", "therapeutic", "activity"],
    "作用": ["effect", "activity", "function"],
    "功效": ["effect", "activity", "therapeutic"],
    "毒性": ["toxicity", "toxic effect", "adverse effect"],
    "副作用": ["side effect", "adverse effect", "toxicity"],
    "药代": ["pharmacokinetics", "metabolism", "bioavailability"],
    "吸收": ["absorption", "bioavailability"],
    "代谢": ["metabolism", "metabolite"],
    "来源": ["source", "isolated from", "derived from"],
    "植物": ["plant", "species", "botanical"],
    "疾病": ["disease", "disorder", "associated with", "indication"]
}


RELATION_EXPANSIONS = {
    "has_antitumor_activity": ["antitumor", "anticancer", "tumor"],
    "has_pharmacological_effect": ["pharmacological effect", "activity"],
    "has_toxic_effect": ["toxicity", "toxic effect"],
    "has_pharmacokinetic_property": ["pharmacokinetics"],
    "binds_to": ["binding", "binds", "target"],
    "isolated_from": ["isolated from", "source"],
    "extracted_from": ["extracted from", "source"],
    "derived_from": ["derived from", "source"]
}


ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "what", "which", "with"
}


GENERIC_QUESTION_TERMS = {
    "activity", "activities", "activate", "activates", "anticancer",
    "antitumor", "bind", "binding", "cancer", "disease", "diseases",
    "effect", "effects", "function", "functions", "inhibit", "inhibits",
    "mechanism", "mechanisms", "pathway", "pathways", "pharmacological",
    "regulate", "regulates", "regulation", "target", "targets", "therapeutic",
    "toxic", "toxicity", "tumor", "tumour"
}


CANCER_CONTEXT_PATTERN = re.compile(
    r"(?:cancer|tumou?r|carcinom|gliom|glioblast|melanom|lymphom|"
    r"leuk(?:a|e)mi|sarcom|myelom|neoplasm|oncolog|anti[- ]?cancer|anti[- ]?tumor|"
    r"肿瘤|癌|肉瘤|淋巴瘤|白血病)",
    re.IGNORECASE
)
CANCER_OUTCOME_PATTERN = re.compile(
    r"(?:anti[- ]?cancer|anti[- ]?tumor|anti-tumor|cytotoxic|apopt|cell\s+death|"
    r"proliferat|viability|cell\s+cycle|growth\s+inhibit|tumou?r\s+growth|"
    r"tumou?r\s+development|migration|invasion|metastas|angiogenesis|"
    r"autophagy|necro|抗肿瘤|抗癌|细胞毒|凋亡|增殖|迁移|侵袭|转移|血管生成)",
    re.IGNORECASE
)


def _normalize(value):
    return unicodedata.normalize(
        "NFKC",
        str(value or "")
    ).casefold().strip()


def _english_tokens(value):
    return [
        token
        for token in re.findall(
            r"[a-z][a-z0-9]*(?:[-/][a-z0-9]+)*",
            _normalize(value)
        )
        if token not in ENGLISH_STOPWORDS and len(token) > 1
    ]


def _add_term(term_weights, term_sources, term, weight, source):
    normalized = _normalize(term).replace("_", " ")

    if not normalized:
        return

    current = term_weights.get(normalized, 0.0)
    term_weights[normalized] = max(current, float(weight))
    term_sources[normalized].add(source)


def expand_query(
    question,
    compound=None,
    intent=None,
    keywords=None,
    relations=None
):
    """Build weighted retrieval terms from the question and parsed intent."""

    term_weights = {}
    term_sources = defaultdict(set)
    normalized_question = _normalize(question)
    compound_tokens = set(_english_tokens(compound))

    for token in _english_tokens(question):
        if token not in compound_tokens:
            _add_term(
                term_weights,
                term_sources,
                token,
                2.4,
                "question"
            )

    for trigger, expansions in CHINESE_QUERY_EXPANSIONS.items():
        if trigger in normalized_question:
            for term in expansions:
                _add_term(
                    term_weights,
                    term_sources,
                    term,
                    2.0,
                    "question_expansion"
                )

    for term in keywords or []:
        _add_term(
            term_weights,
            term_sources,
            term,
            1.7,
            "intent_keyword"
        )

    for term in INTENT_EXPANSIONS.get(intent, []):
        _add_term(
            term_weights,
            term_sources,
            term,
            1.3,
            "intent_expansion"
        )

    for relation in relations or []:
        relation_text = str(relation).replace("_", " ")
        _add_term(
            term_weights,
            term_sources,
            relation_text,
            0.8,
            "relation"
        )

        for term in RELATION_EXPANSIONS.get(relation, []):
            _add_term(
                term_weights,
                term_sources,
                term,
                1.0,
                "relation_expansion"
            )

    expanded_terms = [
        {
            "term": term,
            "weight": round(weight, 3),
            "sources": sorted(term_sources[term])
        }
        for term, weight in sorted(
            term_weights.items(),
            key=lambda item: (-item[1], item[0])
        )
    ]
    specific_terms = [
        term
        for term in term_weights
        if "question" in term_sources[term]
        and term not in GENERIC_QUESTION_TERMS
    ]

    return {
        "terms": expanded_terms,
        "term_weights": term_weights,
        "specific_terms": sorted(specific_terms)
    }


def _reference_key(evidence):
    for field in ("doi", "pmid", "paper_id", "title"):
        value = _normalize(evidence.get(field))

        if value:
            return (field, value)

    return (
        "evidence",
        _normalize(evidence.get("evidence_id")),
        _normalize(evidence.get("evidence_sentence"))
    )


def _deduplicate(evidence):
    deduplicated = []
    seen = set()

    for item in evidence:
        key = (
            _reference_key(item),
            _normalize(item.get("evidence_sentence"))
        )

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append(item)

    return deduplicated


def _field_term_frequency(text, term):
    text = _normalize(text)
    term = _normalize(term)

    if not text or not term:
        return 0

    if " " in term:
        return text.count(term)

    return _english_tokens(text).count(term)


def _estimated_tokens(item, edge):
    text = " ".join([
        str(item.get("title") or ""),
        str(item.get("evidence_sentence") or ""),
        str(item.get("section") or ""),
        str(edge.get("relation") or ""),
        str(edge.get("tail_name") or "")
    ])
    ascii_count = sum(character.isascii() for character in text)
    non_ascii_count = len(text) - ascii_count
    return max(1, math.ceil(ascii_count / 4) + non_ascii_count)


def _intent_compatible(item, edge, intent):
    if intent not in {"anticancer_activity", "anticancer_mechanism"}:
        return True

    searchable = " ".join([
        str(item.get("title") or ""),
        str(item.get("evidence_sentence") or ""),
        str(edge.get("tail_name") or ""),
        str(edge.get("tail_type") or "")
    ])
    sentence = str(item.get("evidence_sentence") or "")
    units = re.split(r"(?<=[.!?。])\s+", sentence)
    background = re.compile(r"anti[- ]?cancer properties", re.I)
    # Remove background assertions before checking outcomes, including in long abstracts.
    units = [unit for unit in units if not background.search(unit)]
    if not units:
        return False
    if item.get("retrieval_source") in {"external_europe_pmc", "local_passage"}:
        cancer_title = bool(CANCER_CONTEXT_PATTERN.search(str(item.get("title") or "")))
        return any((cancer_title or CANCER_CONTEXT_PATTERN.search(unit))
                   and CANCER_OUTCOME_PATTERN.search(unit) for unit in units)
    searchable = " ".join([str(item.get("title") or ""), *units,
                           str(edge.get("tail_name") or ""), str(edge.get("tail_type") or "")])
    return (
        CANCER_CONTEXT_PATTERN.search(searchable) is not None
        and CANCER_OUTCOME_PATTERN.search(searchable) is not None
    )


def _dynamic_target_count(
    ranked,
    min_evidence,
    max_evidence,
    score_threshold
):
    if not ranked:
        return 0, score_threshold, None

    available = len(ranked)
    minimum = min(min_evidence, available)
    maximum = min(max_evidence, available)
    qualified = [
        item
        for item in ranked[:maximum]
        if item["rerank_score"] >= score_threshold
    ]
    target = max(minimum, len(qualified))
    target = min(target, maximum)
    cutoff_reason = "score_threshold"

    largest_gap = 0.0
    gap_target = None

    for index in range(minimum - 1, target - 1):
        current_score = ranked[index]["rerank_score"]
        next_score = ranked[index + 1]["rerank_score"]
        relative_gap = (
            (current_score - next_score) / max(current_score, 1e-9)
        )

        if relative_gap > largest_gap:
            largest_gap = relative_gap
            gap_target = index + 1

    if gap_target is not None and largest_gap >= 0.20:
        target = gap_target
        cutoff_reason = "score_gap"

    effective_threshold = ranked[target - 1]["rerank_score"]
    return target, effective_threshold, cutoff_reason


@timed("reranking")
def rerank_evidence(
    question,
    evidence,
    edges,
    compound=None,
    intent=None,
    keywords=None,
    relations=None,
    min_evidence=6,
    max_evidence=25,
    token_budget=7000,
    score_threshold=0.45,
    rerank_backend=None
):
    """Filter evidence, then rank by rules or a bounded neural/hybrid stage."""
    from app.neural_reranker import backend_name, rerank_rows
    mode = backend_name(rerank_backend)

    min_evidence = max(1, int(min_evidence))
    max_evidence = max(min_evidence, int(max_evidence))
    token_budget = max(1, int(token_budget))
    score_threshold = max(0.0, min(float(score_threshold), 1.0))
    expanded = expand_query(
        question=question,
        compound=compound,
        intent=intent,
        keywords=keywords,
        relations=relations
    )
    term_weights = expanded["term_weights"]
    applied_score_threshold = score_threshold

    if expanded["specific_terms"]:
        applied_score_threshold = max(
            applied_score_threshold,
            0.55
        )
    deduplicated = _deduplicate(evidence)
    edge_map = {
        source_edge_id: edge
        for edge in edges
        for source_edge_id in (
            edge.get("source_edge_ids") or [edge.get("edge_id")]
        )
        if source_edge_id
    }
    attribution_rejected_count = 0
    attribution_rejection_reasons = {}
    if compound:
        (
            deduplicated,
            attribution_rejected,
            attribution_rejection_reasons
        ) = filter_evidence_by_compound_attribution(
            deduplicated,
            compound,
            intent=intent
        )
        attribution_rejected_count = len(attribution_rejected)

    deduplicated_before_intent_filter = len(deduplicated)
    deduplicated = [
        item
        for item in deduplicated
        if _intent_compatible(
            item,
            edge_map.get(item.get("edge_id"), {}),
            intent
        )
    ]
    intent_filtered_count = (
        deduplicated_before_intent_filter - len(deduplicated)
    )

    if not deduplicated:
        return [], {
            "candidate_count": len(evidence),
            "deduplicated_count": 0,
            "compound_attribution_filtered_count": (
                attribution_rejected_count
            ),
            "compound_attribution_rejection_reasons": (
                attribution_rejection_reasons
            ),
            "intent_filtered_count": intent_filtered_count,
            "selected_count": 0,
            "dynamic_target": 0,
            "score_threshold": applied_score_threshold,
            "base_score_threshold": score_threshold,
            "effective_threshold": None,
            "cutoff_reason": "no_candidates",
            "estimated_tokens": 0,
            "token_budget": token_budget,
            "expanded_terms": expanded["terms"],
            "specific_terms": expanded["specific_terms"],
            "focused_candidate_count": 0,
            "reranker": {"requested_backend": mode, "effective_backend": "rules",
                           "status": "no_candidates", "scored_count": 0}
        }

    document_frequency = Counter()

    for item in deduplicated:
        edge = edge_map.get(item.get("edge_id"), {})
        searchable = " ".join([
            str(item.get("title") or ""),
            str(item.get("evidence_sentence") or ""),
            str(edge.get("relation") or "").replace("_", " "),
            str(edge.get("tail_name") or ""),
            str(edge.get("tail_type") or "")
        ])

        for term in term_weights:
            if _normalize(term) in _normalize(searchable):
                document_frequency[term] += 1

    raw_ranked = []
    document_count = len(deduplicated)

    for original_index, item in enumerate(deduplicated):
        edge = edge_map.get(item.get("edge_id"), {})
        raw_score = 0.0
        matched_terms = []

        for term, query_weight in term_weights.items():
            frequency = (
                1.8 * _field_term_frequency(item.get("title"), term)
                + 3.0 * _field_term_frequency(
                    item.get("evidence_sentence"),
                    term
                )
                + 1.4 * _field_term_frequency(
                    str(edge.get("relation") or "").replace("_", " "),
                    term
                )
                + 1.8 * _field_term_frequency(edge.get("tail_name"), term)
                + 0.5 * _field_term_frequency(edge.get("tail_type"), term)
            )

            if frequency <= 0:
                continue

            matched_terms.append(term)
            frequency = frequency / (1.2 + frequency)
            frequency_in_documents = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0
                + (
                    document_count
                    - frequency_in_documents
                    + 0.5
                ) / (frequency_in_documents + 0.5)
            )
            raw_score += (
                query_weight
                * inverse_document_frequency
                * frequency
            )

        confidence = _normalize(item.get("confidence"))
        confidence_bonus = {
            "high": 0.18,
            "medium": 0.09,
            "low": 0.0
        }.get(confidence, 0.04)
        support_bonus = min(
            math.log1p(float(edge.get("paper_count") or 0)) * 0.04,
            0.18
        )
        raw_score += confidence_bonus + support_bonus
        raw_ranked.append({
            "item": item,
            "edge": edge,
            "raw_score": raw_score,
            "matched_terms": matched_terms,
            "original_index": original_index
        })

    maximum_raw_score = max(
        row["raw_score"]
        for row in raw_ranked
    )
    ranked = []

    for row in raw_ranked:
        normalized_score = (
            row["raw_score"] / maximum_raw_score
            if maximum_raw_score > 0
            else 0.0
        )
        ranked.append({
            **row,
            "rerank_score": round(normalized_score, 6)
        })

    ranked.sort(
        key=lambda row: (
            -row["rerank_score"],
            -float(row["edge"].get("paper_count") or 0),
            row["original_index"]
        )
    )
    focused_candidate_count = 0

    if expanded["specific_terms"]:
        specific_term_set = set(expanded["specific_terms"])
        focused_ranked = [
            row
            for row in ranked
            if specific_term_set.intersection(row["matched_terms"])
        ]
        focused_candidate_count = len(focused_ranked)

        if focused_ranked:
            ranked = focused_ranked

    ranked, neural_stats = rerank_rows(question, ranked, backend=mode)
    target, effective_threshold, cutoff_reason = _dynamic_target_count(
        ranked=ranked,
        min_evidence=min_evidence,
        max_evidence=max_evidence,
        score_threshold=applied_score_threshold
    )
    candidate_window = ranked[:max(target * 3, target)]
    selected_rows = []
    selected_indexes = set()
    paper_counts = Counter()
    edge_counts = Counter()
    estimated_tokens = 0

    def try_select(
        row,
        require_new_edge=False,
        allow_paper_repeat=False
    ):
        nonlocal estimated_tokens

        row_index = row["original_index"]

        if row_index in selected_indexes:
            return False

        paper_key = _reference_key(row["item"])
        edge_key = row["item"].get("edge_id")

        if not allow_paper_repeat and paper_counts[paper_key] >= 2:
            return False

        if require_new_edge and edge_counts[edge_key] > 0:
            return False

        item_tokens = _estimated_tokens(row["item"], row["edge"])

        if selected_rows and estimated_tokens + item_tokens > token_budget:
            return False

        selected_rows.append(row)
        selected_indexes.add(row_index)
        paper_counts[paper_key] += 1
        edge_counts[edge_key] += 1
        estimated_tokens += item_tokens
        return True

    for row in candidate_window:
        if len(selected_rows) >= target:
            break

        try_select(row, require_new_edge=True)

    for row in candidate_window:
        if len(selected_rows) >= target:
            break

        try_select(row)

    if len(selected_rows) < min(min_evidence, len(ranked)):
        for row in ranked:
            if len(selected_rows) >= min(min_evidence, len(ranked)):
                break

            try_select(row, allow_paper_repeat=True)

    selected_rows.sort(
        key=lambda row: (
            -row["rerank_score"],
            row["original_index"]
        )
    )
    selected = []

    for row in selected_rows:
        item = dict(row["item"])
        item["rerank_score"] = row["rerank_score"]
        item["matched_query_terms"] = row["matched_terms"]
        for key in ("rule_score", "neural_score", "neural_rank"):
            if key in row:
                item[key] = row[key]
        item["rerank_backend"] = neural_stats["effective_backend"]
        selected.append(item)

    stats = {
        "candidate_count": len(evidence),
        "deduplicated_count": len(deduplicated),
        "compound_attribution_filtered_count": (
            attribution_rejected_count
        ),
        "compound_attribution_rejection_reasons": (
            attribution_rejection_reasons
        ),
        "intent_filtered_count": intent_filtered_count,
        "selected_count": len(selected),
        "dynamic_target": target,
        "score_threshold": applied_score_threshold,
        "base_score_threshold": score_threshold,
        "effective_threshold": (
            round(effective_threshold, 6)
            if effective_threshold is not None
            else None
        ),
        "cutoff_reason": cutoff_reason,
        "estimated_tokens": estimated_tokens,
        "token_budget": token_budget,
        "expanded_terms": expanded["terms"],
        "specific_terms": expanded["specific_terms"],
        "focused_candidate_count": focused_candidate_count,
        "reranker": neural_stats
    }
    return selected, stats

"""Compact reranked evidence into a small, diverse generation context."""

import re
import unicodedata


EFFECT_PATTERNS = [
    ("anti-inflammatory", r"anti[- ]?inflamm|inflammation|抗炎"),
    ("antioxidant", r"anti[- ]?oxid|oxidative stress|抗氧化"),
    ("antimicrobial", r"anti[- ]?(?:microbial|bacterial|fungal)|"
     r"antibacterial|antifungal|抗菌|抗微生物|抗真菌"),
    ("anticancer", r"anti[- ]?(?:cancer|tumou?r)|cancer|tumou?r|"
     r"carcinoma|抗癌|抗肿瘤"),
    ("antidiabetic", r"anti[- ]?diabet|diabetes|抗糖尿病|降糖"),
    ("neuroprotective", r"neuroprotect|神经保护"),
    ("cardioprotective", r"cardioprotect|心脏保护"),
    ("antiviral", r"anti[- ]?viral|抗病毒"),
    ("immunomodulatory", r"immunomodulat|immune|免疫调节"),
]

OBSERVED_RESULT = re.compile(
    r"\b(?:showed|demonstrated|exhibited|inhibited|reduced|increased|"
    r"induced|suppressed|improved|attenuated|downregulat(?:ed|es)|"
    r"upregulat(?:ed|es)|restored|prevented|potentiat(?:ed|es)|"
    r"blocked|activated|regulated|controlled|scaveng(?:ed|es))\b",
    re.IGNORECASE
)
NON_RESULT_EVIDENCE = re.compile(
    r"\b(?:source of support\s*:\s*nil|for determination of|"
    r"was dissolved in|transferred to a .{0,30} flask|"
    r"structural annotation|putative .{0,80} detected|"
    r"identified as (?:a )?(?:constituent|component|marker)|"
    r"patterns? of variation|correlation between|associated with the "
    r"accumulation)\b",
    re.IGNORECASE
)
INDIRECT_STUDY = re.compile(
    r"\b(?:review|perspective|overview|potential|promising|known to|"
    r"has been reported|is reported)\b",
    re.IGNORECASE
)
FORMULATION = re.compile(
    r"\b(?:nano(?:particle|conjugate|emulsion|medicine)?|encapsulat|"
    r"formulat|complex(?:es)?|conjugate(?:s)?|phytosom|liposome)\b",
    re.IGNORECASE
)
COMBINATION_RESULT = re.compile(
    r"\b(?:combination|combined with|co-?administration|plus)\b",
    re.IGNORECASE
)
IDENTITY_EVIDENCE = re.compile(
    r"\b(?:classified as|defined as|pro[- ]?drug|"
    r"selective .{0,20} inhibitor|drug class|coxib)\b|"
    r"(?:是一种|属于|前药|抑制剂|药物类别)",
    re.IGNORECASE
)


def _normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    value = re.sub(r"[^0-9a-z㐀-鿿]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _estimated_tokens(items):
    text = " ".join(
        " ".join([
            str(item.get("title") or ""),
            str(item.get("evidence_sentence") or "")
        ])
        for item in items
    )
    ascii_count = sum(character.isascii() for character in text)
    return max(0, round(ascii_count / 4 + len(text) - ascii_count))


def _edge_map(edges):
    return {
        source_id: edge
        for edge in edges
        for source_id in (
            edge.get("source_edge_ids") or [edge.get("edge_id")]
        )
        if source_id
    }


def _quality(item):
    text = " ".join([
        str(item.get("title") or ""),
        str(item.get("evidence_sentence") or "")
    ])
    score = float(item.get("rerank_score") or 0.0)
    if str(item.get("confidence") or "").casefold() == "high":
        score += 0.12
    if OBSERVED_RESULT.search(str(item.get("evidence_sentence") or "")):
        score += 0.22
    if INDIRECT_STUDY.search(text):
        score -= 0.10
    if FORMULATION.search(text):
        score -= 0.18
    if str(item.get("section") or "").upper() in {"RESULTS", "ABSTRACT"}:
        score += 0.06
    if str(item.get("section") or "").upper().startswith("DISCUSS"):
        score -= 0.08
    return score


def _effect_key(item, edge):
    target = str(edge.get("tail_name") or "")
    context = " ".join([
        str(item.get("evidence_sentence") or ""),
        str(item.get("title") or "")
    ])
    for label, pattern in EFFECT_PATTERNS:
        if re.search(pattern, target, flags=re.IGNORECASE):
            return label
    for label, pattern in EFFECT_PATTERNS:
        if re.search(pattern, context, flags=re.IGNORECASE):
            return label
    return _normalize(target) or _normalize(item.get("title"))


def _source_key(item, edge):
    plant = edge.get("plant_entity") or {}
    value = (
        plant.get("scientific_name")
        or plant.get("canonical_name")
        or edge.get("tail_name")
    )
    return _normalize(value)


def is_direct_intent_evidence(intent, item, edge=None):
    """Whether one row can directly support an answer for ``intent``.

    Ranking relevance alone is deliberately insufficient here: a matching
    paper title, a methods sentence, or a constituent-detection sentence may
    rank highly without supporting the claim that the compound has an effect.
    This predicate is used both by Agent gating and final context compaction.
    """

    edge = edge or {}
    sentence = str(item.get("evidence_sentence") or "").strip()
    title = str(item.get("title") or "").strip()
    if not sentence:
        return False

    if intent == "plant_source":
        if edge.get("relation") not in {
            "isolated_from", "extracted_from", "derived_from",
            "produced_by", "biosynthesized_in", "contains"
        }:
            return False
        plant = edge.get("plant_entity") or {}
        name = str(
            plant.get("scientific_name")
            or plant.get("canonical_name")
            or edge.get("tail_name")
            or ""
        ).strip()
        searchable = f"{title} {sentence}".casefold()
        return bool(name and name.casefold() in searchable)

    if intent == "pharmacological_effect":
        if NON_RESULT_EVIDENCE.search(sentence):
            return False
        context = f"{title} {sentence}"
        if FORMULATION.search(context) or COMBINATION_RESULT.search(context):
            return False
        # The result must occur in the evidence text itself. A strong title
        # cannot turn an unrelated metadata/methods sentence into evidence.
        return OBSERVED_RESULT.search(sentence) is not None

    return True


def _diverse_take(ranked, key_function, limit, fill=False):
    selected = []
    used_keys = set()

    for item, edge in ranked:
        key = key_function(item, edge)
        if key in used_keys:
            continue
        selected.append(item)
        used_keys.add(key)
        if len(selected) >= limit:
            return selected

    if fill:
        for item, _edge in ranked:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected


def compact_intent_evidence(intent, evidence, edges):
    """Return representative evidence and auditable compaction statistics."""

    evidence = list(evidence or [])
    edge_by_id = _edge_map(edges)
    ranked = sorted(
        (
            (item, edge_by_id.get(item.get("edge_id"), {}))
            for item in evidence
        ),
        key=lambda pair: -_quality(pair[0])
    )

    if intent == "plant_source":
        direct_ranked = [
            pair for pair in ranked
            if is_direct_intent_evidence(intent, pair[0], pair[1])
        ]
        selected = _diverse_take(direct_ranked, _source_key, 3, fill=True)
        strategy = "representative_plant_sources"
    elif intent == "pharmacological_effect":
        direct_ranked = [
            pair for pair in ranked
            if is_direct_intent_evidence(intent, pair[0], pair[1])
        ]
        selected = _diverse_take(direct_ranked, _effect_key, 6)
        strategy = "distinct_pharmacological_effects"
    elif intent == "compound_identity":
        identity_ranked = [
            pair for pair in ranked
            if IDENTITY_EVIDENCE.search(" ".join([
                str(pair[0].get("title") or ""),
                str(pair[0].get("evidence_sentence") or "")
            ]))
        ]
        selected = [
            item for item, _edge in (identity_ranked or ranked)[:3]
        ]
        strategy = "representative_compound_identity"
    else:
        selected = evidence
        strategy = "not_compacted"

    return selected, {
        "strategy": strategy,
        "before_count": len(evidence),
        "after_count": len(selected),
        "removed_count": len(evidence) - len(selected),
        "before_estimated_tokens": _estimated_tokens(evidence),
        "after_estimated_tokens": _estimated_tokens(selected)
    }

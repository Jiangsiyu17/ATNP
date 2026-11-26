import json
import re
import hashlib
from collections import Counter
from app.prompt_builder import INTENT_SECTION_LABELS

from app.citation_validator import (
    CITATION_PATTERN,
    normalize_citation_format
)


VALID_VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported"
}
POST_SENTENCE_CITATION_PATTERN = re.compile(
    r"([。！？.!?])\s*((?:\[R\d+\])+)",
    re.IGNORECASE
)
LIST_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[-+*]\s+|\d+\s*[.、)]\s*)"
)
MARKDOWN_PATTERN = re.compile(r"[*_`#]+")
IDENTITY_CLAIM_PATTERN = re.compile(
    r"(?:前药|原药|活性代谢物|非活性前药|"
    r"属于.{0,40}(?:化合物|药物|抑制剂|激动剂|拮抗剂)|"
    r"是一种.{0,40}(?:化合物|药物|抑制剂|激动剂|拮抗剂)|"
    r"\bpro[- ]?drug\b|"
    r"\bis\s+(?:an?\s+)?[^.。]{0,60}"
    r"(?:compound|drug|inhibitor|agonist|antagonist)\b)",
    re.IGNORECASE
)
PLANT_SOURCE_CLAIM_PATTERN = re.compile(
    r"(?:植物来源|来源于|来自|分离自|提取自|纯化自|纯化于|产生于|根茎|果皮|"
    r"\b(?:rhizomes?|roots?|leaves|bark|seeds?|peel)\b|"
    r"isolated from|extracted from|purified\b.{0,80}\bfrom|derived from|plant source)",
    re.IGNORECASE
)
PHARMACOLOGICAL_CLAIM_PATTERN = re.compile(
    r"(?:能够|可(?:以)?|显示|表现|促进|降低|减少|调节|"
    r"镇痛|抗炎|抗氧化|抗菌|抗癌|抗肿瘤|抑制增殖|抑制转移|"
    r"showed|demonstrated|reduced|increased|regulated|"
    r"analgesi|anti[- ]?inflamm|antioxid|anticancer|antitumou?r|"
    r"immunomodulat|免疫调节|hepatoprotect|neuroprotect|"
    r"\b(?:activity|activities|effects?)\b)",
    re.IGNORECASE
)
ACTIVITY_GUIDED_SOURCE_PATTERN = re.compile(
    r"(?:activity[- ]guided|活性(?:导向|指导))",
    re.IGNORECASE
)


def _unique_in_order(values):
    return list(dict.fromkeys(values))


def _infer_claim_intent(claim, eligible_intents, requested_intents, section_intent=None):
    """Assign one supported claim to one requested evidence section."""

    eligible = set(eligible_intents or []) & set(requested_intents or [])
    text = str(claim or "")

    if (
        "compound_identity" in eligible
        and IDENTITY_CLAIM_PATTERN.search(text)
    ):
        return "compound_identity"
    source = bool(PLANT_SOURCE_CLAIM_PATTERN.search(text))
    effect = bool(PHARMACOLOGICAL_CLAIM_PATTERN.search(text))
    if source and effect:
        # "activity-guided purification from Plant X" describes how the
        # compound was isolated.  The word "activity" alone must not move
        # that botanical-origin claim into the pharmacology section.
        if ACTIVITY_GUIDED_SOURCE_PATTERN.search(text):
            return "plant_source" if "plant_source" in eligible else None
        return None  # Mixed source/effect claims need narrowing, not arbitrary placement.
    if source:
        return "plant_source" if "plant_source" in eligible else None
    if effect:
        # Preserve a more specific effect/mechanism section when one was supplied.
        if section_intent in eligible and section_intent not in {"plant_source", "compound_identity"}:
            return section_intent
        if "pharmacological_effect" in eligible:
            return "pharmacological_effect"
        return None
    if section_intent in eligible and section_intent not in {"plant_source", "compound_identity"}:
        return section_intent
    # A shared paper does not determine the intent of an individual statement.
    # In particular, never default an unclassified statement to plant source.
    if len(eligible) == 1 and not eligible & {"plant_source", "compound_identity"}:
        return next(iter(eligible))
    return None


def _claim_mentions_supported_plant(claim, supporting_reference_ids, reference_map):
    """Recognize a cited botanical label without requiring a repeated verb.

    Deterministic source answers intentionally render ``Plant source`` once as
    the section heading and list only plant labels below it.  A label is valid
    only when it repeats a plant name carried by a supporting source relation;
    arbitrary noun phrases therefore do not inherit the section intent.
    """

    text = str(claim or "").casefold()
    for reference_id in supporting_reference_ids or []:
        reference = reference_map.get(reference_id, {})
        for evidence in reference.get("evidence", []):
            for relation in evidence.get("relations", []):
                if relation.get("relation") not in {
                    "isolated_from", "extracted_from", "derived_from",
                    "produced_by", "biosynthesized_in", "contains"
                }:
                    continue
                plant = relation.get("plant_entity") or {}
                names = {
                    str(plant.get("canonical_name") or "").strip(),
                    str(plant.get("scientific_name") or "").strip(),
                    str(relation.get("target") or "").strip(),
                    *(str(name).strip() for name in plant.get("raw_names", []))
                }
                if any(name and name.casefold() in text for name in names):
                    return True
    return False


def _split_claim_sentences(line):
    """Keep botanical authorities/cultivars and parenthesized names intact."""
    parts = []
    start = 0
    depth = 0
    for index, character in enumerate(line):
        if character in "(（[":
            depth += 1
        elif character in ")）]":
            depth = max(0, depth - 1)
        if depth:
            continue
        if character in "。！？!?":
            boundary = True
        elif character == "." and index + 1 < len(line) and line[index + 1].isspace():
            # Taxonomic abbreviations and author initials are not sentence ends.
            prefix = line[:index + 1]
            boundary = not re.search(
                r"(?:\b(?:cv|var|subsp|ssp|sp|spp|cf|aff)|\b[A-Z])\.$",
                prefix
            )
        else:
            boundary = False
        if boundary:
            parts.append(line[start:index + 1])
            start = index + 1
    parts.append(line[start:])
    return parts


def extract_claims(answer):
    """
    从 Markdown 回答中提取可校验的句子及其 [R] 引用。

    这是保守的格式分析：既会提取有引用的结论，也会保留
    没有引用的陈述句，以便将其标记为 unsupported。
    """

    normalized = normalize_citation_format(answer)
    normalized = POST_SENTENCE_CITATION_PATTERN.sub(
        r"\2\1",
        normalized
    )

    claims = []
    section_intent = None
    section_names = {label.casefold(): intent for intent, labels in INTENT_SECTION_LABELS.items()
                     for label in labels.values()}

    for line in normalized.splitlines():
        line = line.strip()

        if not line:
            continue

        line = LIST_PREFIX_PATTERN.sub("", line)
        heading = MARKDOWN_PATTERN.sub("", line).strip().rstrip(":：").casefold()
        if heading in section_names and not CITATION_PATTERN.search(line):
            section_intent = section_names[heading]
            continue  # Only exact structural labels, never arbitrary factual headings.
        if re.match(r"^#{1,6}\s", line):
            section_intent = None

        # 中文句子按句末标点分割；英文句子在句号后有空格时分割。
        sentences = _split_claim_sentences(line)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            citation_ids = _unique_in_order(
                CITATION_PATTERN.findall(sentence)
            )
            claim_text = CITATION_PATTERN.sub("", sentence)
            claim_text = MARKDOWN_PATTERN.sub("", claim_text)
            claim_text = claim_text.strip()

            # 跳过纯标题和过短的非陈述片段。
            if not claim_text:
                continue
            if len(claim_text) < 4:
                continue

            claims.append({
                "claim_id": f"C{len(claims) + 1}",
                "claim": claim_text,
                "citations": citation_ids,
                "section_intent": section_intent
            })

    return claims


def build_claim_validation_prompt(claims, references, compound=None):
    """构建一次性判定多条 claim 的 Strict Evidence Judge Prompt。"""

    reference_map = {
        reference["id"]: reference
        for reference in references
    }
    items = []

    for claim in claims:
        cited_references = []

        for reference_id in claim["citations"]:
            reference = reference_map.get(reference_id)

            if not reference:
                continue

            cited_references.append({
                "id": reference_id,
                "title": reference.get("title"),
                "evidence": [
                    {
                        "evidence_id": item.get("evidence_id"),
                        "sentence": item.get("sentence"),
                        "relation": item.get("relation"),
                        "target": item.get("target")
                    }
                    for item in reference.get("evidence", [])
                ]
            })

        items.append({
            "claim_id": claim["claim_id"],
            "claim": claim["claim"],
            "section_intent": claim.get("section_intent"),
            "cited_references": cited_references
        })

    payload = json.dumps(
        items,
        ensure_ascii=False,
        indent=2
    )

    main_compound = str(compound or "not supplied")

    return f"""You are a strict scientific evidence judge.

Evaluate each claim using ONLY the evidence included with that claim. Do not use outside knowledge.

Verdicts:
- supported: every substantive detail in the claim is directly supported by at least one cited evidence item.
- partially_supported: some details are supported, but the claim adds, merges, generalizes, or overstates details not established by the evidence.
- unsupported: the cited evidence does not establish the claim or contradicts it.

Important rules:
1. Different diseases, cancer types, cell lines, organisms, formulations, experimental models, and evidence types are not interchangeable.
2. A review, prediction, docking result, cell experiment, animal experiment, and clinical result must not be presented as another evidence type.
3. A result for an encapsulated, nanoparticle, conjugated, or otherwise formulated compound does not establish the same result for the free compound.
4. Evidence from different papers must not be fused into one causal mechanism when their models, interventions, outcomes, or effect directions differ.
5. A paper's discussion of an earlier study is secondary evidence and must not be described as an experiment performed in the current paper.
6. The supporting_reference_ids must be a subset of the references supplied for that claim.
7. Return JSON only, without Markdown fences or additional commentary.
8. Give a short reason in the same language as the claim.
9. The main queried compound is {main_compound}. An organizing KG relation is metadata, not proof that a study tested this compound.
10. A compound-specific claim is supported only when the evidence sentence or paper title explicitly identifies the main compound or an unambiguous synonym. Results for a different compound, a numbered compound without an explicit identity mapping, a derivative, an analogue, or a related compound must not be transferred to the main compound.
11. Evaluate EVERY cited reference independently. Put an identifier in supporting_reference_ids only if that specific reference directly supports the claim. Put every other cited identifier in unsupported_reference_ids. One good reference must not hide an unrelated extra citation.
12. An analytical value expressed as "compound equivalent" is not evidence that the compound itself was present or active. A compound detected as an extract constituent or correlated chemical marker does not establish that the isolated compound caused the extract's activity.

Required JSON schema:
{{
  "claims": [
    {{
      "claim_id": "C1",
      "verdict": "supported|partially_supported|unsupported",
      "supporting_reference_ids": ["R1"],
      "unsupported_reference_ids": [],
      "reason": "short reason"
    }}
  ]
}}

Claims and cited evidence:
{payload}
"""


def _parse_judge_response(response_text):
    text = str(response_text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("claim judge did not return a JSON object")

    data = json.loads(text[start:end + 1])
    rows = data.get("claims")

    if not isinstance(rows, list):
        raise ValueError("claim judge JSON must contain a claims list")

    return rows


def _claim_reuse_key(claim):
    """Return an exact key for reusing a verdict after answer repair.

    Reuse is deliberately conservative: wording, citations, and section must
    all stay unchanged.  A narrowed or otherwise edited statement is sent to
    the judge again.
    """

    text = re.sub(r"\s+", " ", str(claim.get("claim") or "")).strip()
    citations = tuple(claim.get("citations") or [])
    return text, citations, claim.get("section_intent")


def _validation_context_signature(references, compound):
    """Bind reused verdicts to the exact compound and evidence context."""

    payload = {
        "compound": str(compound or ""),
        "references": [{
            "id": reference.get("id"),
            "evidence": [{
                "evidence_id": item.get("evidence_id"),
                "sentence": item.get("sentence"),
                "relation": item.get("relation"),
                "target": item.get("target")
            } for item in reference.get("evidence", [])]
        } for reference in references]
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_answer_claims(
    answer,
    references,
    judge,
    compound=None,
    previous_validation=None
):
    """
    对回答中的每条结论做引用覆盖与证据含义校验。

    judge 是一个接收 prompt 并返回文本的可调用对象，通常由
    qa_pipeline 使用本地 LLM 实现。
    """

    claims = extract_claims(answer)
    reference_map = {
        reference["id"]: reference
        for reference in references
    }
    context_signature = _validation_context_signature(references, compound)
    reusable_previous_validation = (
        previous_validation
        if (previous_validation or {}).get("context_signature")
        == context_signature
        else {}
    )
    results_by_id = {}
    judgeable_claims = []
    previous_results = {
        _claim_reuse_key(item): item
        for item in reusable_previous_validation.get("claims", [])
        if item.get("verdict") in VALID_VERDICTS
    }
    reused_claim_count = 0

    for claim in claims:
        citation_ids = claim["citations"]
        known_ids = [
            reference_id
            for reference_id in citation_ids
            if reference_id in reference_map
        ]

        if not citation_ids:
            results_by_id[claim["claim_id"]] = {
                **claim,
                "verdict": "unsupported",
                "supporting_reference_ids": [],
                "unsupported_reference_ids": [],
                "citation_only_failure": False,
                "reason": "该结论没有提供参考文献引用。"
            }
            continue

        if len(known_ids) != len(citation_ids):
            results_by_id[claim["claim_id"]] = {
                **claim,
                "verdict": "unsupported",
                "supporting_reference_ids": [],
                "unsupported_reference_ids": citation_ids,
                "citation_only_failure": False,
                "reason": "该结论包含未知的参考文献编号。"
            }
            continue

        has_evidence = any(
            reference_map[reference_id].get("evidence")
            for reference_id in known_ids
        )

        if not has_evidence:
            results_by_id[claim["claim_id"]] = {
                **claim,
                "verdict": "unsupported",
                "supporting_reference_ids": [],
                "unsupported_reference_ids": citation_ids,
                "citation_only_failure": False,
                "reason": "被引用的参考文献没有可用证据文本。"
            }
            continue

        previous = previous_results.get(_claim_reuse_key(claim))
        if previous is not None:
            # Keep the current claim ID and structural fields.  Section
            # placement is recalculated below against the current references.
            results_by_id[claim["claim_id"]] = {
                **previous,
                **claim,
                "supporting_reference_ids": [
                    reference_id
                    for reference_id in previous.get(
                        "supporting_reference_ids", []
                    )
                    if reference_id in known_ids
                ],
                "unsupported_reference_ids": [
                    reference_id
                    for reference_id in previous.get(
                        "unsupported_reference_ids", []
                    )
                    if reference_id in known_ids
                ],
                "reused_validation": True
            }
            reused_claim_count += 1
            continue

        judgeable_claims.append(claim)

    raw_judge_response = None
    judge_error = None

    if judgeable_claims:
        prompt = build_claim_validation_prompt(
            claims=judgeable_claims,
            references=references,
            compound=compound
        )

        try:
            raw_judge_response = judge(prompt)
            judge_rows = _parse_judge_response(
                raw_judge_response
            )
            judge_map = {
                str(row.get("claim_id")): row
                for row in judge_rows
                if isinstance(row, dict)
            }

            for claim in judgeable_claims:
                row = judge_map.get(claim["claim_id"])

                if not row:
                    results_by_id[claim["claim_id"]] = {
                        **claim,
                        "verdict": "unverified",
                        "supporting_reference_ids": [],
                        "unsupported_reference_ids": claim["citations"],
                        "citation_only_failure": False,
                        "reason": "校验模型未返回该结论的判定。"
                    }
                    continue

                verdict = str(
                    row.get("verdict") or ""
                ).strip().lower()

                if verdict not in VALID_VERDICTS:
                    verdict = "unverified"

                allowed_supporting_ids = set(
                    claim["citations"]
                )
                supporting_ids = _unique_in_order([
                    str(reference_id).upper()
                    for reference_id in (
                        row.get("supporting_reference_ids") or []
                    )
                    if str(reference_id).upper()
                    in allowed_supporting_ids
                ])

                if verdict == "supported" and not supporting_ids:
                    verdict = "unverified"

                unsupported_ids = [
                    reference_id
                    for reference_id in claim["citations"]
                    if reference_id not in supporting_ids
                ]

                # A scientifically true statement still has invalid citation
                # hygiene when one of its attached references is unrelated.
                # Force one repair pass so the model removes only the bad ID.
                if verdict == "supported" and unsupported_ids:
                    verdict = "partially_supported"
                    citation_only_failure = True
                    suffix = (
                        "以下引用未单独支持该结论："
                        if re.search(r"[\u3400-\u9fff]", claim["claim"])
                        else "These citations do not independently support "
                        "the claim: "
                    )
                    reason = str(row.get("reason") or "").strip()
                    reason = " ".join(filter(None, [
                        reason,
                        suffix + ", ".join(unsupported_ids)
                    ]))
                else:
                    reason = str(row.get("reason") or "").strip()
                    citation_only_failure = False

                results_by_id[claim["claim_id"]] = {
                    **claim,
                    "verdict": verdict,
                    "supporting_reference_ids": supporting_ids,
                    "unsupported_reference_ids": unsupported_ids,
                    "citation_only_failure": citation_only_failure,
                    "reason": reason
                }

        except Exception as exc:
            judge_error = f"{type(exc).__name__}: {exc}"

            for claim in judgeable_claims:
                results_by_id[claim["claim_id"]] = {
                    **claim,
                    "verdict": "unverified",
                    "supporting_reference_ids": [],
                    "unsupported_reference_ids": claim["citations"],
                    "citation_only_failure": False,
                    "reason": "证据校验模型返回了无法解析的结果。"
                }

    results = [
        results_by_id[claim["claim_id"]]
        for claim in claims
    ]
    # Evidence support and section placement are separate validations. This
    # adds no model call and allows supported claims to be regrouped locally.
    for result in results:
        section = result.get("section_intent")
        eligible = {intent for rid in result.get("supporting_reference_ids", [])
                    for intent in reference_map.get(rid, {}).get("intent_groups", [])}
        if section and eligible and (result["verdict"] == "supported" or result.get("citation_only_failure")):
            assigned = _infer_claim_intent(result["claim"], eligible, eligible, section)
            if (assigned is None and section == "plant_source"
                    and "plant_source" in eligible
                    and _claim_mentions_supported_plant(
                        result["claim"],
                        result.get("supporting_reference_ids", []),
                        reference_map
                    )):
                assigned = "plant_source"
            result["assigned_intent"] = assigned
            result["section_valid"] = assigned == section
            if assigned != section:
                result["section_reason"] = "Claim does not match its section; regroup or omit if ambiguous."
    counts = Counter(
        result["verdict"]
        for result in results
    )

    return {
        "valid": bool(results) and all(
            result["verdict"] == "supported" and result.get("section_valid", True)
            for result in results
        ),
        "total_claims": len(results),
        "supported_claims": counts["supported"],
        "partially_supported_claims": counts["partially_supported"],
        "unsupported_claims": counts["unsupported"],
        "unverified_claims": counts["unverified"],
        "claims": results,
        "context_signature": context_signature,
        "reused_claims": reused_claim_count,
        "judged_claims": len(judgeable_claims),
        "judge_error": judge_error,
        "raw_judge_response": raw_judge_response
    }


def build_supported_claims_answer(
    validation,
    references=None,
    requested_intents=None,
    section_labels=None,
    response_language=None
):
    """在模型修复仍失败时，只保留已验证的声明。"""

    supported_rows = []
    kept_claim_ids = []
    removed_claim_ids = []

    for item in validation.get("claims", []):
        can_keep = (
            item.get("verdict") == "supported"
            or item.get("citation_only_failure") is True
        )
        reference_ids = list(dict.fromkeys(
            item.get("supporting_reference_ids") or []
        ))
        claim = str(item.get("claim") or "").strip()
        if item.get("section_valid") is False and len(requested_intents or []) <= 1:
            can_keep = can_keep and item.get("assigned_intent") in (requested_intents or [])

        if not can_keep or not reference_ids or not claim:
            removed_claim_ids.append(item.get("claim_id"))
            continue

        punctuation = ""
        if claim.endswith(("。", "！", "？", ".", "!", "?")):
            punctuation = claim[-1]
            claim = claim[:-1].rstrip()
        citation_text = "".join(
            f"[{reference_id}]" for reference_id in reference_ids
        )
        supported_rows.append({
            "claim_id": item.get("claim_id"),
            "claim": claim,
            "reference_ids": reference_ids,
            "section_intent": item.get("section_intent"),
            "line": f"- {claim} {citation_text}{punctuation or '。'}"
        })
        kept_claim_ids.append(item.get("claim_id"))

    intents = list(dict.fromkeys(requested_intents or []))
    if len(intents) <= 1:
        lines = [row["line"] for row in supported_rows]
    else:
        reference_intents = {
            reference["id"]: set(reference.get("intent_groups") or [])
            for reference in (references or [])
        }
        assigned = set()
        lines = []
        is_chinese = str(response_language or "").startswith("zh")

        for row in supported_rows:
            eligible_intents = {
                intent
                for reference_id in row["reference_ids"]
                for intent in reference_intents.get(reference_id, set())
            }
            row["assigned_intent"] = _infer_claim_intent(
                row["claim"],
                eligible_intents,
                intents,
                row.get("section_intent")
            )

        for intent in intents:
            label = (section_labels or {}).get(
                intent,
                str(intent).replace("_", " ").title()
            )
            lines.append(f"### {label}")
            eligible_ids = {
                reference_id
                for reference_id, groups in reference_intents.items()
                if intent in groups
            }
            intent_rows = [
                row for row in supported_rows
                if row["claim_id"] not in assigned
                and row.get("assigned_intent") == intent
                and eligible_ids.intersection(row["reference_ids"])
            ]

            if intent_rows:
                lines.extend(row["line"] for row in intent_rows)
                assigned.update(row["claim_id"] for row in intent_rows)
            elif not eligible_ids:
                lines.append(
                    "当前 ATNP 检索证据不足以回答这一部分。"
                    if is_chinese else
                    "The current ATNP retrieval has insufficient evidence to answer this part."
                )
            else:
                lines.append(
                    "声明核验后，没有保留足够证据支持这一部分。"
                    if is_chinese else
                    "After claim validation, insufficient supported evidence remained for this part."
                )

        unassigned_rows = [
            row for row in supported_rows
            if row["claim_id"] not in assigned
        ]
        if unassigned_rows:
            # Never append ambiguous statements under the last section.
            unassigned_ids = {row["claim_id"] for row in unassigned_rows}
            removed_claim_ids.extend(row["claim_id"] for row in unassigned_rows)
            kept_claim_ids = [cid for cid in kept_claim_ids if cid not in unassigned_ids]
        if not assigned:
            lines = []

    return {
        "answer": "\n".join(lines) if lines else None,
        "kept_claim_ids": kept_claim_ids,
        "claim_sections": {row["claim_id"]: row.get("assigned_intent", row.get("section_intent"))
                           for row in supported_rows if row["claim_id"] in kept_claim_ids},
        "removed_claim_ids": removed_claim_ids
    }

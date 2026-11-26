import html
import re


CITATION_PATTERN = re.compile(r"\[((?:R|E)\d+)\]")
GROUPED_CITATION_PATTERN = re.compile(
    r"\[\s*(?:文献支持\s*[:：]\s*)?"
    r"((?:R|E)\d+(?:\s*[,，、]\s*(?:R|E)\d+)*)\s*\]",
    re.IGNORECASE
)
REFERENCE_ID_PATTERN = re.compile(
    r"(?:R|E)\d+",
    re.IGNORECASE
)
FORBIDDEN_CITATION_PATTERN = re.compile(
    r"\[\s*((?:K\s*\d+)|(?:\d+))\s*\]",
    re.IGNORECASE
)
BOLD_CITATION_PATTERN = re.compile(
    r"\*\*((?:\[R\d+\])+?)\*\*",
    re.IGNORECASE
)


def decode_html_entities(value):
    text = str(value or "")

    # Handle the occasional double-encoded entity without unbounded decoding.
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded

    return text


def normalize_citation_format(answer):
    """
    将模型常见的引用格式统一为 [R1][R2] 格式。

    例如：
    [文献支持：R1] -> [R1]
    [文献支持：R1, R2] -> [R1][R2]
    [R1，R2] -> [R1][R2]
    """

    def replace_group(match):
        reference_ids = REFERENCE_ID_PATTERN.findall(
            match.group(1)
        )

        return "".join(
            f"[{reference_id.upper()}]"
            for reference_id in reference_ids
        )

    answer = decode_html_entities(answer)
    answer = BOLD_CITATION_PATTERN.sub(r"\1", answer)
    return GROUPED_CITATION_PATTERN.sub(
        replace_group,
        answer
    )


def validate_citations(answer, references):
    """
    检查模型输出中的引用编号，并移除未提供给模型的引用。

    这个校验只能防止伪造引用编号，不能代替后续的 claim-level
    faithfulness 评估。
    """

    original_answer = str(answer or "")
    decoded_answer = decode_html_entities(original_answer)
    html_entities_decoded = decoded_answer != original_answer
    answer = normalize_citation_format(original_answer)
    forbidden_ids = [
        re.sub(r"\s+", "", match).upper()
        for match in FORBIDDEN_CITATION_PATTERN.findall(answer)
    ]
    answer = FORBIDDEN_CITATION_PATTERN.sub("", answer)
    allowed_ids = {
        reference["id"]
        for reference in references
    }
    found_ids = set(CITATION_PATTERN.findall(answer))
    invalid_ids = found_ids - allowed_ids

    def remove_invalid(match):
        citation_id = match.group(1)
        return "" if citation_id in invalid_ids else match.group(0)

    cleaned_answer = CITATION_PATTERN.sub(
        remove_invalid,
        answer
    )

    cited_ids = [
        reference["id"]
        for reference in references
        if reference["id"] in found_ids
    ]
    invalid_ids = sorted(set(invalid_ids) | set(forbidden_ids))
    missing_citations = bool(
        cleaned_answer.strip()
        and references
        and not cited_ids
    )

    cited_references = [
        reference
        for reference in references
        if reference["id"] in cited_ids
    ]

    return {
        "answer": cleaned_answer,
        "valid": not invalid_ids and not missing_citations,
        "has_citations": bool(cited_ids),
        "missing_citations": missing_citations,
        "cited_reference_ids": cited_ids,
        "invalid_reference_ids": invalid_ids,
        "forbidden_citation_ids": sorted(set(forbidden_ids)),
        "html_entities_decoded": html_entities_decoded,
        "cited_references": cited_references
    }

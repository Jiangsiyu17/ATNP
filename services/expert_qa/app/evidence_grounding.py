"""High-precision attribution checks for compound-specific evidence."""

import re
from collections import Counter

from app.entity_resolver import (
    get_unambiguous_compound_terms,
    normalize_entity_term
)


DERIVATIVE_AFTER_PATTERN = re.compile(
    r"^\s*(?:[-\u2010-\u2015]\s*)?(?:"
    r"(?:derived|based|related|derivatives?|analog(?:ue)?s?|conjugates?|"
    r"precursors?|metabolites?)\b|"
    r"(?:and\b.{0,100}\b(?:based\s+)?derivatives?\b)|"
    r"(?:\u884d\u751f\u7269|\u7c7b\u4f3c\u7269|\u7c7b\u4f3c\u4f53|\u5076\u8054\u7269|\u524d\u4f53|\u4ee3\u8c22\u7269)"
    r")",
    re.IGNORECASE
)
DERIVATIVE_BEFORE_PATTERN = re.compile(
    r"(?:derivatives?|analog(?:ue)?s?|conjugates?|precursors?|metabolites?)"
    r"\s+of\s*$",
    re.IGNORECASE
)
CHEMICAL_LOCANT_BEFORE_PATTERN = re.compile(
    r"(?:\b\d+(?:,\d+)*(?:['′\"]*)-(?:o|n|s|c)-|"
    r"\b(?:di|tri|tetra)methyl-)\s*$",
    re.IGNORECASE
)
# A canonical parent/scaffold name can also occur as the terminal part of a
# longer systematic name.  The ordinary token-boundary check deliberately
# permits hyphens because they are meaningful inside chemical names, so it
# would otherwise attribute e.g. ``...-2-(...)-4H-chromen-4-one`` to the
# standalone entity ``4H-chromen-4-one``.  Require the hyphen to be directly
# attached on both sides; this does not reject a standalone name following a
# list dash (``- 4H-chromen-4-one``).
EMBEDDED_CHEMICAL_PREFIX_PATTERN = re.compile(
    r"(?:[0-9a-z\)\]\}′'])-$",
    re.IGNORECASE
)

BIOLOGICAL_EFFECT_INTENTS = {
    "anticancer_activity",
    "anticancer_mechanism",
    "disease_association",
    "pharmacokinetics",
    "pharmacological_effect",
    "target_mechanism",
    "toxicity"
}
ANALYTICAL_EQUIVALENT_PATTERN = re.compile(
    r"^\s*(?:[-\u2010-\u2015]\s*)?equivalents?\b",
    re.IGNORECASE
)
EXTRACT_AFTER_PATTERN = re.compile(
    r"^\s*(?:[-\u2010-\u2015]\s*)?rich\b.{0,60}"
    r"\b(?:extract|fraction|mixture|preparation)\b",
    re.IGNORECASE
)
COMPONENT_BEFORE_PATTERN = re.compile(
    r"(?:\b(?:including|containing|contains|contained|comprising|"
    r"constituents?|components?|markers?|identification|identified|detected|profiled)\b"
    r"|(?:\u5305\u62ec|\u542b\u6709|\u6210\u5206|\u6807\u5fd7\u7269|\u9274\u5b9a|\u68c0\u51fa))"
    r".{0,90}$",
    re.IGNORECASE
)
COMPONENT_AFTER_PATTERN = re.compile(
    r"^.{0,120}(?:\b(?:constituents?|components?|chemical\s+markers?|"
    r"identification|identified|detected|profiled)\b|(?:\u6210\u5206|\u5316\u5b66\u6807\u5fd7\u7269|\u9274\u5b9a|\u68c0\u51fa))",
    re.IGNORECASE
)
FAMILY_AFTER_PATTERN = re.compile(
    r"^.{0,45}\b(?:member|representative)\b.{0,70}"
    r"\b(?:family|series|class|group)\b.{0,100}"
    r"\b(?:known|reported|possess|including)\b",
    re.IGNORECASE
)
FAMILY_BEFORE_PATTERN = re.compile(
    r"\b(?:family|series|class|group)\b.{0,100}"
    r"\b(?:including|such\s+as)\s*$",
    re.IGNORECASE
)
STUDY_DESIGN_PATTERN = re.compile(
    r"\b(?:was|were)\s+(?:tested|evaluated|assessed|investigated)\b"
    r"|\b(?:we\s+)?(?:tested|evaluated|assessed|investigated)\b",
    re.IGNORECASE
)
OBSERVED_RESULT_PATTERN = re.compile(
    r"\b(?:showed|demonstrated|exhibited|resulted|caused|induced|"
    r"promoted|modulated|regulated|activated|suppressed|inhibited|"
    r"reduced|decreased|increased|was\s+active|were\s+active|"
    r"was\s+inactive|were\s+inactive|had\s+no\s+activity|"
    r"significantly|potent)\b",
    re.IGNORECASE
)


def _usable_term(term):
    term = normalize_entity_term(term)
    if not term or term.isdigit() or len(term) > 160:
        return False

    # One- and two-letter Latin aliases cause many accidental matches in
    # biomedical abstracts. Chinese names remain useful from two characters.
    if term.isascii() and len(term) < 3:
        return False

    return True


def _term_matches(text, term):
    normalized_text = normalize_entity_term(text)
    normalized_term = normalize_entity_term(term)
    if not normalized_text or not normalized_term:
        return []

    if normalized_term.isascii():
        pattern = re.compile(
            rf"(?<![0-9a-z]){re.escape(normalized_term)}(?![0-9a-z])",
            re.IGNORECASE
        )
        return list(pattern.finditer(normalized_text))

    return list(re.finditer(re.escape(normalized_term), normalized_text))


def _derivative_only_mention(text, match):
    before = text[max(0, match.start() - 45):match.start()]
    after = text[match.end():match.end() + 45]
    return bool(
        DERIVATIVE_AFTER_PATTERN.search(after)
        or DERIVATIVE_BEFORE_PATTERN.search(before)
        # Avoid matching the suffix of a different substituted compound,
        # e.g. treating "methyl 3-O-methyl gallate" as methyl gallate.
        or CHEMICAL_LOCANT_BEFORE_PATTERN.search(before)
        # Avoid treating a parent/scaffold suffix inside a longer systematic
        # chemical name as the standalone queried compound.
        or EMBEDDED_CHEMICAL_PREFIX_PATTERN.search(before)
    )


def _non_agent_role(text, match, intent):
    """Return a hard rejection reason for a non-intervention mention."""

    before = text[max(0, match.start() - 140):match.start()]
    after = text[match.end():match.end() + 180]

    # ``µg olivetol equivalent/g`` means that olivetol was a calibration
    # standard. It does not show that olivetol was present or bioactive.
    if ANALYTICAL_EQUIVALENT_PATTERN.search(after):
        return "analytical_equivalent_only"

    if intent not in BIOLOGICAL_EFFECT_INTENTS:
        return None

    # Activity of an olivetol-rich extract is activity of a mixture, not of
    # isolated/free olivetol.
    if EXTRACT_AFTER_PATTERN.search(after):
        return "extract_or_mixture_only"

    # Metabolomics/component lists establish correlation or occurrence only.
    # They must not be converted into a pharmacological effect of the named
    # compound.
    if (
        COMPONENT_BEFORE_PATTERN.search(before)
        and COMPONENT_AFTER_PATTERN.search(after)
    ):
        return "constituent_or_marker_only"

    # A property stated for an entire chemical family does not automatically
    # apply to every named member of that family.
    if (
        FAMILY_AFTER_PATTERN.search(after)
        or FAMILY_BEFORE_PATTERN.search(before)
    ):
        return "compound_family_only"

    # A methods-only sentence saying that compounds were tested does not
    # report which one worked. The KG edge/title must not fill in that result.
    if (
        STUDY_DESIGN_PATTERN.search(text)
        and not OBSERVED_RESULT_PATTERN.search(text)
    ):
        return "study_design_only"

    return None


def _field_attribution(text, terms, intent=None):
    normalized_text = normalize_entity_term(text)
    derivative_terms = []
    non_agent_modes = []

    for term in terms:
        matches = _term_matches(normalized_text, term)
        if not matches:
            continue

        direct_matches = []
        for match in matches:
            if _derivative_only_mention(normalized_text, match):
                continue

            non_agent_mode = _non_agent_role(
                normalized_text,
                match,
                intent
            )
            if non_agent_mode:
                non_agent_modes.append(non_agent_mode)
                continue

            direct_matches.append(match)

        if direct_matches:
            return {
                "matched": True,
                "derivative_only": False,
                "non_agent_mode": None,
                "matched_term": term
            }
        if any(
            _derivative_only_mention(normalized_text, match)
            for match in matches
        ):
            derivative_terms.append(term)

    if non_agent_modes:
        return {
            "matched": False,
            "derivative_only": False,
            "non_agent_mode": non_agent_modes[0],
            "matched_term": None
        }

    if derivative_terms:
        return {
            "matched": False,
            "derivative_only": True,
            "non_agent_mode": None,
            "matched_term": derivative_terms[0]
        }

    return {
        "matched": False,
        "derivative_only": False,
        "non_agent_mode": None,
        "matched_term": None
    }


def assess_compound_attribution(
    evidence,
    compound,
    intent=None
):
    """Assess whether one evidence record can be attributed to ``compound``.

    The KG edge itself is intentionally not considered evidence of identity;
    it is precisely the field that can be incorrectly linked upstream.
    """

    terms = [
        term
        for term in get_unambiguous_compound_terms(compound)
        if _usable_term(term)
    ]
    if not terms and _usable_term(compound):
        terms = [normalize_entity_term(compound)]

    if not terms:
        return {
            "accepted": False,
            "mode": "unverifiable_compound",
            "matched_term": None
        }

    sentence = _field_attribution(
        evidence.get("evidence_sentence"),
        terms,
        intent=intent
    )
    if sentence["matched"]:
        return {
            "accepted": True,
            "mode": "evidence_sentence",
            "matched_term": sentence["matched_term"]
        }
    if sentence["derivative_only"]:
        return {
            "accepted": False,
            "mode": "derivative_only",
            "matched_term": sentence["matched_term"]
        }
    if sentence["non_agent_mode"]:
        return {
            "accepted": False,
            "mode": sentence["non_agent_mode"],
            "matched_term": sentence["matched_term"]
        }

    title = _field_attribution(
        evidence.get("title"),
        terms,
        intent=intent
    )
    if title["matched"]:
        return {
            "accepted": True,
            "mode": "paper_title",
            "matched_term": title["matched_term"]
        }
    if title["derivative_only"]:
        return {
            "accepted": False,
            "mode": "derivative_only",
            "matched_term": title["matched_term"]
        }
    if title["non_agent_mode"]:
        return {
            "accepted": False,
            "mode": title["non_agent_mode"],
            "matched_term": title["matched_term"]
        }

    return {
        "accepted": False,
        "mode": "compound_not_mentioned",
        "matched_term": None
    }


def filter_evidence_by_compound_attribution(
    evidence,
    compound,
    intent=None
):
    """Keep only evidence whose text can be tied to the queried compound."""

    kept = []
    rejected = []
    reasons = Counter()

    for item in evidence:
        attribution = assess_compound_attribution(
            item,
            compound,
            intent=intent
        )
        annotated = dict(item)
        annotated["compound_attribution"] = attribution

        if attribution["accepted"]:
            kept.append(annotated)
        else:
            rejected.append(annotated)
            reasons[attribution["mode"]] += 1

    return kept, rejected, dict(sorted(reasons.items()))

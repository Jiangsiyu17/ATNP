"""High-precision relation/type constraints for noisy KG extractions."""

import re


RELATION_ALLOWED_TAIL_TYPES = {
    # Disease/pathogen entities mislabeled as antitumor activity are a common
    # extraction error. Keep only explicit cancer concepts and cancer cell
    # lines for this relation.
    "has_antitumor_activity": {"Cancer", "Cell_Line"},
    "isolated_from": {"Plant", "Plant_Tissue", "Organism"},
    "extracted_from": {"Plant", "Plant_Tissue", "Organism", "Food"},
    "produced_by": {"Plant", "Plant_Tissue", "Organism"},
    "biosynthesized_in": {"Plant", "Plant_Tissue", "Organism", "Tissue"}
}


ANTITUMOR_FALSE_ENTITY_PATTERN = re.compile(
    r"(?:plasmodium|candida|dengue|chikungunya|leishmani|"
    r"bacterial\s+wilt|malaria|bacter(?:ia|ium)|virus|fung(?:us|al))",
    re.IGNORECASE
)


def is_relation_target_allowed(relation, tail_type, tail_name=None):
    allowed_types = RELATION_ALLOWED_TAIL_TYPES.get(str(relation or ""))

    if allowed_types is None:
        return True

    if str(tail_type or "") not in allowed_types:
        return False

    if (
        str(relation or "") == "has_antitumor_activity"
        and ANTITUMOR_FALSE_ENTITY_PATTERN.search(str(tail_name or ""))
    ):
        return False

    return True


def filter_relation_targets(edges):
    kept = []
    rejected = []

    for edge in edges:
        if is_relation_target_allowed(
            edge.get("relation"),
            edge.get("tail_type"),
            edge.get("tail_name")
        ):
            kept.append(edge)
        else:
            rejected.append(edge)

    return kept, rejected

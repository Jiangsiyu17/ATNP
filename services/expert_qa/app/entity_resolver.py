import csv
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

import duckdb

from app.runtime_paths import DATA_DIR, KG_DB_PATH


DB_PATH = KG_DB_PATH
ALIAS_PATH = DATA_DIR / "compound_aliases.csv"
EXTERNAL_ALIAS_PATH = (
    DATA_DIR / "compound_aliases_external.csv"
)


# These Chinese names are genuinely ambiguous and must never silently resolve
# to whichever shorter alias happens to match first.
CURATED_AMBIGUOUS_ALIASES = {
    "甲基没食子酸": [
        "Methyl gallate",
        "3-O-Methylgallic acid"
    ]
}


_COMPOUND_NAME_INDEX = None


def normalize_entity_term(value):
    """Normalize lookup text without deleting chemically meaningful locants."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _contains_term(text, term):
    if not term or term not in text:
        return False

    # Paper-local numeric labels cannot identify a compound in free text.
    if not any(character.isalpha() for character in term):
        return False

    if not re.search(r"[\u3400-\u9fff]", term):
        # Hyphens and locants belong to chemical names, including Greek prefixes.
        pattern = rf"(?<![0-9a-zα-ω_\-′']){re.escape(term)}(?![0-9a-zα-ω_\-′'])"
        return re.search(pattern, text) is not None

    return term in text


def _empty_candidate(canonical_name, canonical_key):
    return {
        "canonical_name": canonical_name,
        "canonical_key": canonical_key,
        "node_ids": set(),
        "display_options": [],
        "terms": defaultdict(set)
    }


def _candidate_public(candidate, matched_term=None, sources=None):
    node_ids = sorted(candidate["node_ids"])
    return {
        "canonical_name": candidate["canonical_name"],
        "compound_node_id": node_ids[0] if node_ids else None,
        "compound_node_ids": node_ids,
        "matched_term": matched_term,
        "match_sources": sorted(sources or [])
    }


def load_compound_name_index(force_reload=False):
    """
    Build one full-database compound index from canonical names, nodes.aliases,
    and the manually curated CSV aliases.

    A normalized term maps to a set of canonical candidates. This is
    intentionally one-to-many so alias collisions become clarification results
    instead of arbitrary matches.
    """

    global _COMPOUND_NAME_INDEX

    if _COMPOUND_NAME_INDEX is not None and not force_reload:
        return _COMPOUND_NAME_INDEX

    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT
                n.node_id,
                n.name,
                n.aliases,
                coalesce(e.edge_count, 0) AS edge_count
            FROM nodes n
            LEFT JOIN (
                SELECT head_id, count(*) AS edge_count
                FROM edges
                GROUP BY head_id
            ) e ON e.head_id = n.node_id
            WHERE n.type = 'Compound'
              AND n.name IS NOT NULL
              AND trim(n.name) <> ''
            ORDER BY lower(trim(n.name)), edge_count DESC, n.name
            """
        ).fetchall()
    finally:
        connection.close()

    compounds = {}
    term_candidates = defaultdict(dict)

    def ensure_candidate(canonical_name):
        canonical_key = normalize_entity_term(canonical_name)
        candidate = compounds.get(canonical_key)

        if candidate is None:
            candidate = _empty_candidate(canonical_name, canonical_key)
            compounds[canonical_key] = candidate

        return candidate

    def add_term(candidate, term, source):
        normalized_term = normalize_entity_term(term)
        if not normalized_term:
            return

        candidate["terms"][normalized_term].add(source)
        term_candidates[normalized_term][candidate["canonical_key"]] = (
            candidate
        )

    for node_id, name, aliases, edge_count in rows:
        candidate = ensure_candidate(name)
        candidate["node_ids"].add(str(node_id))
        candidate["display_options"].append((name, int(edge_count or 0)))
        add_term(candidate, name, "canonical")

        for alias in aliases or []:
            add_term(candidate, alias, "db_alias")

    for candidate in compounds.values():
        candidate["canonical_name"] = max(
            candidate["display_options"],
            key=lambda item: (
                not item[0].islower(),
                not item[0].isupper(),
                item[1],
                -len(item[0])
            )
        )[0]

    for alias_path, source_name in (
        (ALIAS_PATH, "manual_alias"),
        (EXTERNAL_ALIAS_PATH, "external_alias")
    ):
        if not alias_path.exists():
            continue

        with alias_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                status = str(row.get("status") or "approved").strip().lower()
                if status not in {"approved", "auto_approved"}:
                    continue

                manual_canonical_name = str(
                    row.get("canonical_name") or ""
                ).strip()
                canonical_key = normalize_entity_term(
                    manual_canonical_name
                )
                candidate = compounds.get(canonical_key)

                # A typo in the manual alias table must not invent an entity
                # that does not exist in the knowledge graph.
                if candidate is None:
                    continue

                # Only the curated table may choose the preferred display
                # case. Generated external rows can contain a lower-case
                # duplicate such as ``curcumin`` and must add aliases only.
                if source_name == "manual_alias":
                    candidate["canonical_name"] = manual_canonical_name
                add_term(candidate, row.get("alias"), source_name)

    for alias, canonical_names in CURATED_AMBIGUOUS_ALIASES.items():
        for canonical_name in canonical_names:
            candidate = compounds.get(normalize_entity_term(canonical_name))
            if candidate is not None:
                add_term(candidate, alias, "curated_ambiguous_alias")

    terms = sorted(term_candidates, key=len, reverse=True)
    _COMPOUND_NAME_INDEX = {
        "compounds": compounds,
        "term_candidates": term_candidates,
        "terms": terms,
        "stats": {
            "canonical_entities": len(compounds),
            "canonical_nodes": sum(
                len(candidate["node_ids"])
                for candidate in compounds.values()
            ),
            "indexed_terms": len(term_candidates)
        }
    }
    return _COMPOUND_NAME_INDEX


def resolve_compound_name(value):
    """Resolve one exact canonical name or alias to node IDs."""

    normalized = normalize_entity_term(value)
    candidates = list(
        load_compound_name_index()["term_candidates"].get(
            normalized,
            {}
        ).values()
    )
    return _resolution_from_candidates(
        mention=str(value or "").strip(),
        normalized_term=normalized,
        candidates=candidates
    )


def get_unambiguous_compound_terms(value):
    """Return verified names that uniquely identify one compound entity.

    The evidence-grounding layer uses these terms to verify that a paper title
    or evidence sentence actually discusses the queried compound.  Ambiguous
    aliases are deliberately excluded: seeing an ambiguous Chinese name in a
    paper must not silently bind that evidence to one arbitrary KG node.
    """

    resolution = resolve_compound_name(value)
    if resolution["status"] != "resolved":
        return []

    index = load_compound_name_index()
    canonical_key = normalize_entity_term(
        resolution["canonical_name"]
    )
    candidate = index["compounds"].get(canonical_key)
    if candidate is None:
        return [resolution["canonical_name"]]

    terms = []
    for term in candidate["terms"]:
        owners = index["term_candidates"].get(term, {})
        if len(owners) == 1 and canonical_key in owners:
            terms.append(term)

    return sorted(
        set(terms),
        key=lambda term: (-len(term), term)
    )


def _resolution_from_candidates(mention, normalized_term, candidates):
    unique = {
        candidate["canonical_key"]: candidate
        for candidate in candidates
    }
    curated_order = {
        normalize_entity_term(alias): {
            normalize_entity_term(name): index
            for index, name in enumerate(names)
        }
        for alias, names in CURATED_AMBIGUOUS_ALIASES.items()
    }.get(normalized_term, {})
    candidates = sorted(
        unique.values(),
        key=lambda item: (
            curated_order.get(item["canonical_key"], 10_000),
            item["canonical_name"].casefold()
        )
    )

    if not candidates:
        return {
            "status": "not_found",
            "mention": mention,
            "candidates": []
        }

    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "mention": mention,
            "candidates": [
                _candidate_public(
                    candidate,
                    matched_term=mention,
                    sources=candidate["terms"].get(normalized_term, set())
                )
                for candidate in candidates
            ]
        }

    candidate = candidates[0]
    return {
        "status": "resolved",
        "mention": mention,
        **_candidate_public(
            candidate,
            matched_term=mention,
            sources=candidate["terms"].get(normalized_term, set())
        )
    }


def resolve_compound_in_text(text):
    """Resolve all longest explicit compound mentions in a user question."""

    normalized_text = normalize_entity_term(text)
    index = load_compound_name_index()
    matches = []

    for term in index["terms"]:
        if _contains_term(normalized_text, term):
            matches.append(term)

    if not matches:
        return {
            "status": "not_found",
            "mention": None,
            "candidates": []
        }

    # Remove a shorter alias contained within a longer matched alias. This is
    # what prevents 没食子酸 from stealing 甲基没食子酸.
    maximal_terms = [
        term
        for term in matches
        if not any(
            term != other and term in other
            for other in matches
        )
    ]
    candidates = []

    for term in maximal_terms:
        candidates.extend(index["term_candidates"][term].values())

    mention = " / ".join(maximal_terms)
    resolution = _resolution_from_candidates(
        mention=mention,
        normalized_term=(maximal_terms[0] if len(maximal_terms) == 1 else ""),
        candidates=candidates
    )
    resolution["matched_terms"] = maximal_terms
    return resolution


def suggest_compound_entities(query, limit=10):
    normalized_query = normalize_entity_term(query)
    if not normalized_query:
        return []

    limit = max(1, min(int(limit), 50))
    index = load_compound_name_index()
    ranked = {}

    for term, candidate_map in index["term_candidates"].items():
        if normalized_query == term:
            score = 0
        elif term.startswith(normalized_query):
            score = 1
        elif normalized_query in term:
            score = 2
        else:
            continue

        for candidate in candidate_map.values():
            key = candidate["canonical_key"]
            row = {
                "name": candidate["canonical_name"],
                "compound_node_id": (
                    sorted(candidate["node_ids"])[0]
                    if candidate["node_ids"] else None
                ),
                "compound_node_ids": sorted(candidate["node_ids"]),
                "matched_text": term,
                "matched_alias": (
                    term
                    if "canonical" not in candidate["terms"].get(term, set())
                    else None
                ),
                "match_sources": sorted(candidate["terms"].get(term, set())),
                "_score": score
            }
            current = ranked.get(key)
            current_key = (
                current["_score"], len(current["matched_text"])
            ) if current else None
            row_key = (row["_score"], len(row["matched_text"]))

            if current is None or row_key < current_key:
                ranked[key] = row

    rows = sorted(
        ranked.values(),
        key=lambda row: (
            row["_score"],
            len(row["matched_text"]),
            row["name"].casefold()
        )
    )[:limit]

    for row in rows:
        row.pop("_score", None)

    return rows


def _levenshtein_distance(left, right):
    """Small Unicode-safe edit distance used only for suggestions."""

    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (
                    left_character != right_character
                )
            ))
        previous = current
    return previous[-1]


def suggest_similar_compound_entities(query, limit=5):
    """
    Suggest close Chinese aliases or Latin names for explicit confirmation.

    This is intentionally separate from ``resolve_compound_name``: a typo may
    display “did you mean”, but it must never choose a compound silently.
    """

    normalized_query = normalize_entity_term(query)
    if not normalized_query:
        return []
    is_chinese = bool(re.search(r"[\u3400-\u9fff]", normalized_query))
    if not is_chinese and (
        len(normalized_query) < 6
        or not any(character.isalpha() for character in normalized_query)
    ):
        return []

    index = load_compound_name_index()
    query_length = len(normalized_query)
    if is_chinese:
        max_distance = 1 if query_length <= 3 else 2
    else:
        max_distance = 2 if query_length < 12 else 3
    ranked = {}

    for term, candidate_map in index["term_candidates"].items():
        if bool(re.search(r"[\u3400-\u9fff]", term)) != is_chinese:
            continue
        suffix_candidate = (
            not is_chinese and len(term) >= 6
            and normalized_query.startswith(term + "-")
            and re.fullmatch(r"[a-z0-9]{1,8}", normalized_query[len(term) + 1:]) is not None
        )
        if not suffix_candidate and abs(len(term) - query_length) > max_distance:
            continue

        distance = _levenshtein_distance(normalized_query, term)
        if distance == 0 or (not suffix_candidate and distance > max_distance):
            continue

        similarity = SequenceMatcher(
            None,
            normalized_query,
            term
        ).ratio()

        # Two edits in a four-character drug name are accepted only as a
        # confirmation candidate; they are never treated as an exact alias.
        if not suffix_candidate and similarity < (0.45 if is_chinese else 0.85):
            continue

        for candidate in candidate_map.values():
            key = candidate["canonical_key"]
            row = {
                **_candidate_public(
                    candidate,
                    matched_term=term,
                    sources=candidate["terms"].get(term, set())
                ),
                "suggested_alias": term,
                "requires_suffix_confirmation": suffix_candidate,
                "unverified_suffix": normalized_query[len(term) + 1:] if suffix_candidate else None,
                "edit_distance": distance,
                "similarity": round(similarity, 4)
            }
            rank = (distance, -similarity, len(term))
            if key not in ranked or rank < ranked[key][0]:
                ranked[key] = (rank, row)

    ordered = sorted(ranked.values(), key=lambda item: item[0])
    if not ordered:
        return []

    best_distance = ordered[0][0][0]
    return [
        item[1]
        for item in ordered
        if item[0][0] == best_distance
    ][:limit]

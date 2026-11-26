import re
import unicodedata


SOURCE_RELATIONS = {
    "isolated_from",
    "extracted_from",
    "derived_from",
    "produced_by",
    "biosynthesized_in",
    "contains"
}

PLANT_PART_PATTERNS = [
    (r"^stem\s+barks?\s+of\s+", "stem bark"),
    (r"^root\s+barks?\s+of\s+", "root bark"),
    (r"^barks?\s+of\s+", "bark"),
    (r"^rhizomes?\s+of\s+", "rhizome"),
    (r"^roots?\s+of\s+", "root"),
    (r"^leaves?\s+of\s+", "leaf"),
    (r"^flowers?\s+of\s+", "flower"),
    (r"^fruits?\s+of\s+", "fruit"),
    (r"^seeds?\s+of\s+", "seed"),
]

TRAILING_PART_PATTERN = re.compile(
    r"\s+(stem bark|root bark|bark|rhizome|root|leaf|leaves|flower|fruit|seed)s?$",
    re.IGNORECASE
)

# Conservative list of forms found in botanical author citations. Unknown
# third tokens are preserved rather than stripped.
AUTHORITY_PATTERN = re.compile(
    r"^(?:L\.|Gay|Hook\.f\.|Mill\.|DC\.|Roxb\.|Thunb\.|Linn\.)$",
    re.IGNORECASE
)


def normalize_plant_name(value):
    raw_name = unicodedata.normalize("NFKC", str(value or "")).strip()
    working = re.sub(r"\s+", " ", raw_name)
    plant_parts = []

    for pattern, plant_part in PLANT_PART_PATTERNS:
        if re.search(pattern, working, flags=re.IGNORECASE):
            working = re.sub(
                pattern,
                "",
                working,
                count=1,
                flags=re.IGNORECASE
            ).strip()
            plant_parts.append(plant_part)
            break

    trailing = TRAILING_PART_PATTERN.search(working)
    if trailing:
        plant_part = trailing.group(1).casefold()
        if plant_part == "leaves":
            plant_part = "leaf"
        plant_parts.append(plant_part)
        working = working[:trailing.start()].strip()

    tokens = working.split()
    authority = None

    if len(tokens) >= 2:
        genus = tokens[0][:1].upper() + tokens[0][1:].lower()
        species = tokens[1].lower()
        canonical_name = f"{genus} {species}"
        remainder = tokens[2:]

        if remainder and all(AUTHORITY_PATTERN.match(token) for token in remainder):
            authority = " ".join(remainder)
        elif remainder:
            canonical_name = " ".join([canonical_name, *remainder])
    else:
        canonical_name = working

    return {
        "canonical_name": canonical_name,
        "scientific_name": canonical_name,
        "taxonomic_authority": authority,
        "plant_parts": sorted(set(plant_parts)),
        "raw_names": [raw_name]
    }


def normalize_and_merge_plant_edges(edges):
    """Normalize botanical labels and merge duplicate source relations."""

    merged = []
    positions = {}

    for original in edges:
        edge = dict(original)
        edge.setdefault("source_edge_ids", [edge.get("edge_id")])
        is_plant_source = (
            edge.get("relation") in SOURCE_RELATIONS
            and edge.get("tail_type") in {"Plant", "Plant_Tissue"}
        )

        if not is_plant_source:
            merged.append(edge)
            continue

        plant = normalize_plant_name(edge.get("tail_name"))
        key = (
            edge.get("relation"),
            plant["canonical_name"].casefold()
        )
        position = positions.get(key)

        if position is None:
            edge["raw_tail_name"] = edge.get("tail_name")
            edge["tail_name"] = plant["canonical_name"]
            edge["plant_entity"] = plant
            positions[key] = len(merged)
            merged.append(edge)
            continue

        current = merged[position]
        current["source_edge_ids"] = sorted(set(
            current.get("source_edge_ids", [])
            + edge.get("source_edge_ids", [])
        ))
        current_plant = current["plant_entity"]
        current_plant["raw_names"] = sorted(set(
            current_plant.get("raw_names", [])
            + plant.get("raw_names", [])
        ))
        current_plant["plant_parts"] = sorted(set(
            current_plant.get("plant_parts", [])
            + plant.get("plant_parts", [])
        ))
        authorities = {
            value
            for value in (
                current_plant.get("taxonomic_authority"),
                plant.get("taxonomic_authority")
            )
            if value
        }
        current_plant["taxonomic_authority"] = (
            "; ".join(sorted(authorities)) or None
        )
        current["paper_count"] = max(
            int(current.get("paper_count") or 0),
            int(edge.get("paper_count") or 0)
        )
        current["evidence_count"] = (
            int(current.get("evidence_count") or 0)
            + int(edge.get("evidence_count") or 0)
        )

    return merged

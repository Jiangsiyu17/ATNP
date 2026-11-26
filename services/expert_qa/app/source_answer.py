"""Render strict plant-source facts with verbatim botanical labels."""
import re
import unicodedata


PARENTHETICAL_BINOMIAL = re.compile(
    r"\(\s*([A-Za-z][A-Za-z.-]+\s+[a-z][A-Za-z.-]+)\s*\)"
)
STANDALONE_BINOMIAL = re.compile(
    r"^([A-Z][A-Za-z.-]+\s+[a-z][A-Za-z.-]+)$"
)


def _source_key(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _source_aliases(name):
    """Return conservative aliases used only to merge displayed sources.

    A combined label such as ``Turmeric (Curcuma longa)`` bridges the common
    name and scientific name.  We do not use a general taxonomy guess or
    translate names that are absent from evidence.
    """

    aliases = {_source_key(name)}
    match = PARENTHETICAL_BINOMIAL.search(str(name or ""))
    if match:
        outer = str(name)[:match.start()].strip(" ,")
        if outer:
            aliases.add(_source_key(outer))
        aliases.add(_source_key(match.group(1)))
    return {alias for alias in aliases if alias}


def _format_source_name(name):
    """Normalize capitalization inside an evidenced common/scientific label."""

    text = re.sub(r"\s+", " ", str(name or "")).strip()
    match = PARENTHETICAL_BINOMIAL.search(text)
    if not match:
        return text
    genus, species = match.group(1).split(None, 1)
    scientific = genus[:1].upper() + genus[1:].lower() + " " + species.lower()
    outer = text[:match.start()].strip(" ,")
    return f"{outer} ({scientific})" if outer else scientific


def _merge_sources(records):
    """Merge names connected by an explicit common/scientific bridge label."""

    groups = []
    for record in records:
        overlapping = [group for group in groups if group["aliases"] & record["aliases"]]
        if not overlapping:
            groups.append({
                "aliases": set(record["aliases"]),
                "names": [record["name"]],
                "ids": set(record["ids"]),
            })
            continue
        target = overlapping[0]
        target["aliases"].update(record["aliases"])
        target["names"].append(record["name"])
        target["ids"].update(record["ids"])
        for extra in overlapping[1:]:
            target["aliases"].update(extra["aliases"])
            target["names"].extend(extra["names"])
            target["ids"].update(extra["ids"])
            groups.remove(extra)

    for group in groups:
        # Prefer an evidenced label that already contains both common and
        # scientific names; otherwise preserve the original first label.
        combined = [name for name in group["names"] if PARENTHETICAL_BINOMIAL.search(name)]
        scientific = [name for name in group["names"] if STANDALONE_BINOMIAL.match(name)]
        chosen = (combined or scientific or group["names"])[0]
        group["display_name"] = _format_source_name(chosen)
    return groups


def build_source_answer(references, language):
    """Build a source section from relations that are also explicit in text.

    This deliberately uses both the structured ``isolated_from`` relation and
    the cited sentence/title.  It is therefore safe to reuse in a multi-intent
    answer without asking the generation model to reinterpret the source fact.
    References belonging only to other intents are ignored.
    """
    sources = {}
    for reference in references:
        if 'plant_source' not in reference.get('intent_groups', []):
            continue
        for evidence in reference.get('evidence', []):
            if 'plant_source' not in evidence.get('intent_groups', []):
                continue
            for relation in evidence.get('relations', []):
                if relation.get('relation') not in {'isolated_from', 'extracted_from', 'derived_from'}:
                    continue
                plant = relation.get('plant_entity') or {}
                name = str(plant.get('canonical_name') or relation.get('target') or '').strip()
                # Require the label in actual text/title; the KG label alone is not proof.
                text = str(reference.get('title') or '') + ' ' + str(evidence.get('sentence') or '')
                if not name or name.casefold() not in text.casefold():
                    continue
                if re.search(r'[\n\r\[\]<>]', name):
                    continue
                sources.setdefault(name, set()).add(reference['id'])
    if not sources:
        return None
    if language == 'zh':
        lines = ['### 植物来源']
    else:
        lines = ['### Plant source']
    records = [
        {"name": name, "ids": ids, "aliases": _source_aliases(name)}
        for name, ids in sources.items()
    ]
    for source in _merge_sources(records):
        name, ids = source["display_name"], source["ids"]
        citations = ''.join(f'[{identifier}]' for identifier in sorted(ids))
        if language == 'zh':
            lines.append(f'- {name} {citations}。')
        else:
            lines.append(f'- {name} {citations}.')
    return '\n'.join(lines)

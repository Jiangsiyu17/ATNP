"""Read-only, same-paper abbreviation grounding and potential direction conflicts."""
import re
import sqlite3
from app import literature_tools


def condition_definitions(text, terms):
    definitions = {}
    for match in re.finditer(r'([A-Za-z][A-Za-z -]{2,100})\s*\(([A-Z][A-Z0-9-]{1,7})\)', text):
        long, short = match.groups()
        matched = next((term for term in terms if term.isascii() and re.search(
            r'(?<![a-z])' + re.escape(term) + r'\s*$', long, re.I)), None)
        definitions.setdefault(short, []).append((matched, match.group(0)))
    # An abbreviation with competing definitions in this paper is not usable.
    return {short: {'term': values[0][0], 'definition': values[0][1]}
            for short, values in definitions.items() if all(v[0] for v in values)
            and len({v[0].casefold() for v in values}) == 1}


def attach_condition_context(rows, terms, max_papers=64):
    cache, result = {}, []
    index = literature_tools.INDEX_PATH
    connection = sqlite3.connect('file:' + str(index) + '?mode=ro', uri=True) if index.is_file() else None
    try:
        for original in rows:
            row = dict(original)
            text = str(row.get('evidence_sentence') or '')
            # Only inspect papers whose candidate contains a possible acronym.
            if re.search(r'\b[A-Z][A-Z0-9-]{1,7}\b', text):
                key = (row.get('doi'), row.get('pmid'), row.get('paper_id'))
                context = []
                if key not in cache and len(cache) < max_papers:
                    if connection and any(key):
                        try:
                            docs = connection.execute('SELECT document_id FROM documents WHERE lower(doi)=lower(?) OR pmid=? OR paper_id=?', key).fetchall()
                            for (doc,) in docs[:2]:
                                context.extend({'passage_id': 'passage:' + pid, 'text': body}
                                    for pid, body in connection.execute(
                                        'SELECT id,text FROM passages WHERE document_id=? ORDER BY position LIMIT 8', (doc,)))
                        except sqlite3.Error:
                            context = []  # Missing context must not license an expansion.
                    cache[key] = context
                context = cache.get(key, [])
                full = '\n'.join([str(row.get('title') or ''), text, *[c['text'] for c in context]])
                definitions = condition_definitions(full, terms)
                if definitions:
                    row['condition_context'] = {'paper_id': row.get('paper_id'), 'definitions': definitions,
                        'passage_ids': [c['passage_id'] for c in context]}
            result.append(row)
    finally:
        if connection:
            connection.close()
    return result


def direction_conflicts(groups):
    """Flag textual disagreement, not proof of a scientific contradiction."""
    conflicts = []
    verb = re.compile(r'\b(inhibit\w*|suppress\w*|reduc\w*|decreas\w*|induc\w*|promot\w*|increas\w*)\b', re.I)
    for compound, rows in groups.items():
        directions = {'increase': [], 'decrease': []}
        for row in rows:
            text = row['evidence_sentence']
            for endpoint in re.finditer(r'\bapoptosis\b', text, re.I):
                prefix = re.split(r'[.;!?]', text[:endpoint.start()])[-1]
                verbs = list(verb.finditer(prefix))
                if not verbs:
                    continue
                governing = verbs[-1]
                if len(prefix) - governing.end() > 120 or re.search(r'\b(?:not|no|failed|may|might)\b', prefix, re.I):
                    continue
                direction = 'increase' if re.match(r'induc|promot|increas', governing.group(), re.I) else 'decrease'
                if row not in directions[direction]:
                    directions[direction].append(row)
        if all(directions.values()):
            conflicts.append({'compound': compound, 'endpoint': 'apoptosis', 'status': 'potential_conflict',
                'note': 'Different models or interventions may explain the disagreement; original text is unchanged.',
                'evidence': [{**row, 'observed_direction': direction} for direction, values in directions.items() for row in values[:2]]})
    return conflicts


def render_conflicts(conflicts, references, compound, zh):
    ids = {eid: r['id'] for r in references for e in r['evidence'] for eid in e['evidence_ids']}
    lines = []
    for conflict in conflicts:
        if conflict['compound'] != compound:
            continue
        lines.append('凋亡方向存在文献表述差异，待核实实验条件；以下原句不作为一致结论：' if zh else
                     'Potential disagreement in apoptosis direction; check study conditions. The original statements are not a consensus:')
        for row in conflict['evidence']:
            if row['evidence_id'] in ids:
                lines.append(row['evidence_sentence'] + ' [' + ids[row['evidence_id']] + ']')
    return '\n\n'.join(lines)

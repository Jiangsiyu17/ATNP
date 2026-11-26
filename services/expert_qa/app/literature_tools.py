"""Read-only retrieval from the local literature passage index; no web search."""
from app.performance import timed
from contextlib import closing
import json
import os
import logging
import time
import re
import sqlite3
from urllib.parse import quote

from app.entity_resolver import get_unambiguous_compound_terms
from app.request_queue import check_cancelled
from app.runtime_paths import PASSAGE_INDEX_PATH

INDEX_PATH = PASSAGE_INDEX_PATH


def _row(identifier, title, text, **meta):
    return dict(evidence_id=identifier, edge_id=identifier, title=title,
                evidence_sentence=text, confidence='unknown', **meta)


LOGGER = logging.getLogger(__name__)


def passage_index_status():
    if not INDEX_PATH.is_file():
        return {'status': 'not_configured'}
    try:
        with closing(sqlite3.connect('file:' + quote(str(INDEX_PATH)) + '?mode=ro', uri=True, timeout=5)) as con:
            con.execute('SELECT id FROM passages LIMIT 1').fetchone()
            has_manifest = con.execute("SELECT 1 FROM sqlite_master WHERE name='build_info'").fetchone()
            row = con.execute("SELECT value FROM build_info WHERE key='manifest'").fetchone() if has_manifest else None
        manifest = json.loads(row[0]) if row else {}
        return {'status': 'ok', 'schema_version': manifest.get('schema_version', 1),
                'documents': manifest.get('documents'), 'passages': manifest.get('passages'),
                'body_documents': manifest.get('body_documents'),
                'abstract_only_documents': manifest.get('abstract_only_documents'),
                'size_bytes': INDEX_PATH.stat().st_size}
    except (sqlite3.Error, OSError, ValueError) as exc:
        return {'status': 'error', 'error_type': type(exc).__name__}


def _passage_units(row):
    row = dict(row)
    metadata = {key: row.get(key) for key in (
        'paper_id', 'doi', 'pmid', 'pmcid', 'section', 'page', 'source_path',
        'text_source', 'source_archive', 'source_line', 'section_index', 'start_char', 'end_char')}
    if row.get('pmcid'):
        metadata['source_url'] = 'https://pmc.ncbi.nlm.nih.gov/articles/' + quote(row['pmcid'], safe='') + '/'
    return [_row('passage:' + row['id'] + ':' + str(i), row['title'], sentence.strip(),
                 **metadata, source_passage_id='passage:' + row['id'], retrieval_source='local_passage')
            for i, sentence in enumerate(re.split(r'(?<=[.!?。])\s+(?=[A-Z\u3400-\u9fff])', row['text']))
            if sentence.strip()]


def _phrase_terms(terms, maximum=30):
    # Quote every user/model term as an FTS phrase, never as an operator.
    values = list(dict.fromkeys(str(t).strip() for t in terms if str(t).strip()))[:maximum]
    return ['"' + value[:160].replace('"', '""') + '"' for value in values
            if re.search(r'[\w]', value, re.UNICODE)]


def _search_index(query, *, limit=30, offset=0):
    check_cancelled()
    empty = {'edges': [], 'evidence': [], 'passage_count': 0, 'has_more': False}
    if not INDEX_PATH.is_file():
        return {**empty, 'tool_status': 'not_configured'}
    if not query:
        return {**empty, 'tool_status': 'ok'}
    limit, offset = max(1, min(100, int(limit))), max(0, int(offset))
    try:
        with closing(sqlite3.connect('file:' + quote(str(INDEX_PATH)) + '?mode=ro', uri=True, timeout=5)) as con:
            con.row_factory = sqlite3.Row
            deadline = time.monotonic() + 10
            con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            # Rank and truncate in the FTS table before joining large passage text.
            rows = con.execute("""SELECT p.* FROM
                (SELECT rowid, rank FROM passages_fts WHERE passages_fts MATCH ?
                 ORDER BY rank, rowid LIMIT ? OFFSET ?) f
                JOIN passages p ON p.rowid=f.rowid ORDER BY f.rank, f.rowid""",
                (query, limit + 1, offset)).fetchall()
        selected = rows[:limit]
        return {'edges': [], 'evidence': [unit for row in selected for unit in _passage_units(row)],
                'tool_status': 'ok', 'passage_count': len(selected), 'has_more': len(rows) > limit}
    except sqlite3.Error as exc:
        LOGGER.warning('Passage search failed: %s', type(exc).__name__)
        return {**empty, 'tool_status': 'error', 'error_type': type(exc).__name__}


@timed("fulltext_retrieval")
def search_passages(compound, keywords):
    # Do not load the entity index when fulltext has not been configured.
    if not INDEX_PATH.is_file():
        return _search_index('')
    aliases = get_unambiguous_compound_terms(compound)
    terms = [compound] + [a for a in aliases if 3 <= len(a) <= 120][:12]
    identity = _phrase_terms(terms)
    focus = _phrase_terms(keywords, 8)
    query = '(' + ' OR '.join(identity) + ')'
    if focus:
        query += ' AND (' + ' OR '.join(focus) + ')'
    return _search_index(query if identity else '')


@timed("condition_fulltext_retrieval")
def search_condition_passages(terms, *, limit=30, offset=0, require_outcome=False):
    """Disease/function recall has no root compound; attribution happens downstream."""
    phrases = _phrase_terms(terms)
    query = 'text : (' + ' OR '.join(phrases) + ')' if phrases else ''
    if query and require_outcome:
        # Fixed outcome prefixes reduce background-only hits before LIMIT.
        # Sentence-level and semantic checks remain mandatory downstream.
        query += (' AND text : (inhibit* OR suppress* OR reduc* OR decreas* OR restrain* OR block*'
                  ' OR induc* OR promot* OR increas* OR cytotoxic* OR exhibit* OR show*'
                  ' OR demonstrat* OR exert* OR possess* OR ameliorat* OR improv*)')
    return _search_index(query, limit=limit, offset=offset)


@timed("passage_read")
def read_passages(identifiers):
    check_cancelled()
    if not INDEX_PATH.is_file():
        return {'edges': [], 'evidence': [], 'tool_status': 'not_configured'}
    found = {}
    try:
        with closing(sqlite3.connect('file:' + quote(str(INDEX_PATH)) + '?mode=ro', uri=True, timeout=5)) as con:
            con.row_factory = sqlite3.Row
            for identifier in identifiers[:3]:
                key = identifier.removeprefix('passage:').split(':')[0]
                rows = con.execute("""SELECT p.* FROM passages p JOIN passages anchor ON anchor.id=?
                    WHERE p.document_id=anchor.document_id AND p.position BETWEEN anchor.position-1 AND anchor.position+1
                    ORDER BY p.position""", (key,)).fetchall()
                for row in rows:
                    for unit in _passage_units(row):
                        found[unit['evidence_id']] = unit
        return {'edges': [], 'evidence': list(found.values()), 'tool_status': 'ok'}
    except sqlite3.Error as exc:
        return {'edges': [], 'evidence': [], 'tool_status': 'error', 'error_type': type(exc).__name__}

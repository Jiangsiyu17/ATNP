from contextlib import closing
from functools import lru_cache

from .literature import LiteratureDatabaseError, _connect, _like_pattern, _normalise_query


PAGE_SIZE = 20


def get_compound_evidence(compound_name):
    """Return compound relationships, evidence, and their source papers."""
    compound_name = _normalise_query(compound_name)
    if not compound_name:
        return []

    exact = compound_name.casefold()
    fuzzy = _like_pattern(compound_name)
    parameters = [exact, exact, fuzzy, exact, exact, fuzzy, fuzzy]

    try:
        with closing(_connect()) as connection:
            rows = connection.execute(
                """
                WITH candidates AS (
                    SELECT node_id,
                           CASE
                             WHEN lower(name) = ? OR (
                               aliases IS NOT NULL AND
                               list_contains(
                                 list_transform(aliases, alias -> lower(alias)), ?
                               )
                             ) THEN 0
                             WHEN name ILIKE ? ESCAPE '\\' THEN 1
                             ELSE 2
                           END AS match_rank
                    FROM nodes
                    WHERE lower(type) = 'compound'
                      AND (
                           lower(name) = ?
                        OR (aliases IS NOT NULL AND list_contains(
                             list_transform(aliases, alias -> lower(alias)), ?
                           ))
                        OR name ILIKE ? ESCAPE '\\'
                        OR CAST(aliases AS VARCHAR) ILIKE ? ESCAPE '\\'
                      )
                ),
                matched_nodes AS (
                    SELECT node_id FROM candidates
                    WHERE match_rank = (SELECT MIN(match_rank) FROM candidates)
                ),
                matched_edges AS (
                    SELECT DISTINCT e.edge_id, e.head_name, e.head_type,
                                    e.relation, e.tail_name, e.tail_type,
                                    e.evidence_count
                    FROM edges e
                    JOIN matched_nodes n
                      ON e.head_id = n.node_id OR e.tail_id = n.node_id
                )
                SELECT DISTINCT
                       me.edge_id, me.head_name, me.head_type, me.relation,
                       me.tail_name, me.tail_type,
                       ev.evidence_sentence, ev.confidence, ev.section,
                       ev.paper_id, ev.pmid, ev.title, ev.doi, ev.pmcid,
                       me.evidence_count
                FROM matched_edges me
                JOIN evidence ev ON ev.edge_id = me.edge_id
                WHERE NULLIF(TRIM(ev.evidence_sentence), '') IS NOT NULL
                ORDER BY me.evidence_count DESC,
                         CASE lower(ev.confidence)
                           WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2
                         END,
                         me.head_name, me.relation, me.tail_name, ev.title
                """,
                parameters,
            ).fetchall()
    except LiteratureDatabaseError:
        raise
    except Exception as exc:
        raise LiteratureDatabaseError(
            "Related evidence could not be loaded from the knowledge database."
        ) from exc

    return [
        {
            "edge_id": row[0], "head_name": row[1], "head_type": row[2],
            "relation": row[3], "tail_name": row[4], "tail_type": row[5],
            "evidence_sentence": row[6], "confidence": row[7],
            "section": row[8], "paper_id": row[9], "pmid": row[10],
            "title": row[11], "doi": row[12], "pmcid": row[13],
        }
        for row in rows
    ]


@lru_cache(maxsize=1)
def get_relations():
    """Return the relation vocabulary used by the knowledge graph."""
    try:
        with closing(_connect()) as connection:
            return [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT relation FROM edges "
                    "WHERE relation IS NOT NULL AND relation <> '' ORDER BY relation"
                ).fetchall()
            ]
    except LiteratureDatabaseError:
        raise
    except Exception as exc:
        raise LiteratureDatabaseError(
            "The knowledge graph relations could not be loaded."
        ) from exc


def search_knowledge(head="", relation="", tail="", page=1, page_size=PAGE_SIZE):
    """Search graph triples using any one or more supplied fields."""
    head = _normalise_query(head)
    relation = _normalise_query(relation)
    tail = _normalise_query(tail)
    page = max(int(page or 1), 1)
    page_size = min(max(int(page_size), 1), 100)

    if not any((head, relation, tail)):
        return {"triples": [], "total": 0}

    conditions = []
    parameters = []
    if head:
        conditions.append("head_name ILIKE ? ESCAPE '\\'")
        parameters.append(_like_pattern(head))
    if relation:
        conditions.append("relation = ?")
        parameters.append(relation)
    if tail:
        conditions.append("tail_name ILIKE ? ESCAPE '\\'")
        parameters.append(_like_pattern(tail))

    where_sql = " AND ".join(conditions)
    rank_parts = []
    rank_parameters = []
    if head:
        rank_parts.append("CASE WHEN lower(head_name) = lower(?) THEN 0 ELSE 1 END")
        rank_parameters.append(head)
    if tail:
        rank_parts.append("CASE WHEN lower(tail_name) = lower(?) THEN 0 ELSE 1 END")
        rank_parameters.append(tail)
    rank_sql = " + ".join(rank_parts) if rank_parts else "0"
    offset = (page - 1) * page_size

    try:
        with closing(_connect()) as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM edges WHERE {where_sql}", parameters
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT edge_id, head_name, head_type, relation,
                       tail_name, tail_type, occurrence_count,
                       paper_count, evidence_count
                FROM edges
                WHERE {where_sql}
                ORDER BY ({rank_sql}),
                         evidence_count DESC NULLS LAST,
                         paper_count DESC NULLS LAST,
                         occurrence_count DESC NULLS LAST,
                         head_name, tail_name
                LIMIT ? OFFSET ?
                """,
                parameters + rank_parameters + [page_size, offset],
            ).fetchall()

            references = {}
            edge_ids = [row[0] for row in rows]
            if edge_ids:
                placeholders = ", ".join("?" for _ in edge_ids)
                reference_rows = connection.execute(
                    f"""
                    SELECT edge_id, title, pmid, doi, pmcid
                    FROM evidence
                    WHERE edge_id IN ({placeholders})
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY edge_id
                        ORDER BY CASE lower(confidence)
                                   WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2
                                 END,
                                 evidence_id
                    ) = 1
                    """,
                    edge_ids,
                ).fetchall()
                references = {
                    row[0]: {
                        "title": row[1], "pmid": row[2],
                        "doi": row[3], "pmcid": row[4],
                    }
                    for row in reference_rows
                }

    except LiteratureDatabaseError:
        raise
    except Exception as exc:
        raise LiteratureDatabaseError(
            "The knowledge graph search could not be completed."
        ) from exc

    triples = []
    for row in rows:
        triples.append({
            "edge_id": row[0], "head_name": row[1], "head_type": row[2],
            "relation": row[3], "tail_name": row[4], "tail_type": row[5],
            "occurrence_count": row[6], "paper_count": row[7],
            "evidence_count": row[8], "reference": references.get(row[0]),
        })
    return {"triples": triples, "total": int(total)}

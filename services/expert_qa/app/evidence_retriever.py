import duckdb

from app.runtime_paths import KG_DB_PATH


DB_PATH = KG_DB_PATH


def get_evidence_by_edge_ids(
    edge_ids,
    limit=100,
    focus_terms=None
):
    """
    根据 edge_id 列表查询支持证据。
    """

    if not edge_ids:
        return []

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True
    )

    placeholders = ",".join(
        ["?"] * len(edge_ids)
    )

    focus_terms = [
        str(term).strip()
        for term in (focus_terms or [])
        if str(term).strip()
    ][:20]
    focus_order = ""
    focus_params = []
    if focus_terms:
        conditions = " OR ".join(
            "strpos(lower(coalesce(evidence_sentence, '') || ' ' || "
            "coalesce(title, '')), lower(?)) > 0"
            for _term in focus_terms
        )
        focus_order = f"CASE WHEN {conditions} THEN 1 ELSE 0 END DESC,"
        focus_params.extend(focus_terms)

    sql = f"""
    SELECT
        evidence_id,
        edge_id,
        paper_id,
        pmid,
        pmcid,
        doi,
        title,
        section,
        chunk_id,
        evidence_sentence,
        confidence
    FROM evidence
    WHERE edge_id IN ({placeholders})
    ORDER BY
        {focus_order}
        CASE lower(coalesce(confidence, ''))
            WHEN 'high' THEN 3
            WHEN 'medium' THEN 2
            WHEN 'low' THEN 1
            ELSE 0
        END DESC
    LIMIT ?
    """

    params = list(edge_ids) + focus_params + [limit]

    result = con.execute(
        sql,
        params
    )

    columns = [
        desc[0]
        for desc in result.description
    ]

    rows = result.fetchall()

    con.close()

    return [
        dict(zip(columns, row))
        for row in rows
    ]


if __name__ == "__main__":

    from app.kg_retriever import search_compound_knowledge

    edges = search_compound_knowledge(
        compound="Curcumin",
        relations=["regulates"],
        limit=5
    )

    print("Selected edges:")
    print("=" * 100)

    for e in edges:
        print(
            e["relation"],
            "->",
            e["tail_name"],
            "| edge_id:",
            e["edge_id"]
        )

    edge_ids = [
        e["edge_id"]
        for e in edges
    ]

    evidence = get_evidence_by_edge_ids(
        edge_ids,
        limit=30
    )

    print("\nEvidence:")
    print("=" * 100)

    for i, ev in enumerate(evidence, 1):
        print(f"\n[{i}]")
        print("Title:", ev["title"])
        print("DOI:", ev["doi"])
        print("PMID:", ev["pmid"])
        print("Section:", ev["section"])
        print("Confidence:", ev["confidence"])
        print("Evidence:", ev["evidence_sentence"])

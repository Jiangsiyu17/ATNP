import duckdb

from app.entity_resolver import resolve_compound_name
from app.request_queue import check_cancelled
from app.plant_normalizer import normalize_and_merge_plant_edges
from app.relation_policy import (
    RELATION_ALLOWED_TAIL_TYPES,
    filter_relation_targets
)
from app.runtime_paths import KG_DB_PATH


DB_PATH = KG_DB_PATH


def search_compound_knowledge(
    compound: str,
    compound_node_id=None,
    compound_node_ids=None,
    relations=None,
    focus_terms=None,
    limit: int = 50
):
    """
    根据化合物名称查询知识图谱中的关系。

    Parameters
    ----------
    compound : str
        化合物名称，例如 Curcumin

    relations : list[str] | None
        可选，只查询指定关系。
        例如：
        ["inhibits", "regulates", "targets"]

    limit : int
        最多返回多少条 edge

    Returns
    -------
    list[dict]
    """

    check_cancelled()
    con = duckdb.connect(
        str(DB_PATH),
        read_only=True
    )

    resolved_node_ids = [
        str(node_id)
        for node_id in (compound_node_ids or [])
        if str(node_id).strip()
    ]

    if compound_node_id and str(compound_node_id) not in resolved_node_ids:
        resolved_node_ids.append(str(compound_node_id))

    if not resolved_node_ids:
        resolution = resolve_compound_name(compound)
        if resolution["status"] == "resolved":
            resolved_node_ids = resolution["compound_node_ids"]

    sql = """
    SELECT
        edge_id,
        head_id,
        head_name,
        head_type,
        relation,
        tail_id,
        tail_name,
        tail_type,
        paper_count,
        evidence_count
    FROM edges
    WHERE 1 = 1
    """

    params = []

    if resolved_node_ids:
        placeholders = ",".join(["?"] * len(resolved_node_ids))
        sql += f"\nAND head_id IN ({placeholders})\n"
        params.extend(resolved_node_ids)
    else:
        # Compatibility fallback for callers that supply a legacy name not
        # present in the resolver index. Normal application traffic resolves
        # to node IDs before this function is called.
        sql += "\nAND lower(head_name) = lower(?)\n"
        params.append(compound)

    if relations:
        placeholders = ",".join(
            ["?"] * len(relations)
        )

        sql += f"""
        AND relation IN ({placeholders})
        """

        params.extend(relations)

    for relation, allowed_types in RELATION_ALLOWED_TAIL_TYPES.items():
        type_placeholders = ",".join(["?"] * len(allowed_types))
        sql += (
            f"\nAND (relation <> ? OR tail_type IN "
            f"({type_placeholders}))\n"
        )
        params.append(relation)
        params.extend(sorted(allowed_types))

    focus_terms = [
        str(term).strip()
        for term in (focus_terms or [])
        if str(term).strip()
    ]

    if focus_terms:
        focus_conditions = " OR ".join([
            "strpos(lower(replace(tail_name, 'κ', 'k')), lower(?)) > 0"
            for _ in focus_terms
        ])
        sql += f"""
        ORDER BY
            CASE WHEN {focus_conditions} THEN 1 ELSE 0 END DESC,
            paper_count DESC,
            evidence_count DESC
        LIMIT ?
        """
        params.extend(focus_terms)
    else:
        sql += """
        ORDER BY
            paper_count DESC,
            evidence_count DESC
        LIMIT ?
        """

    # Source aliases may merge into one canonical botanical entity after the
    # SQL query. A larger internal window avoids losing diversity after merge.
    params.append(max(int(limit) * 3, int(limit)))

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

    rows = [
        dict(zip(columns, row))
        for row in rows
    ]
    rows, _rejected_rows = filter_relation_targets(rows)
    rows = normalize_and_merge_plant_edges(rows)
    return rows[:limit]


if __name__ == "__main__":

    rows = search_compound_knowledge(
        compound="Curcumin",
        limit=30
    )

    print(
        f"Found {len(rows)} edges for Curcumin"
    )

    print("=" * 100)

    for r in rows:
        print(
            f"{r['head_name']:15s} "
            f"-- {r['relation']:30s} --> "
            f"{r['tail_name'][:40]:40s} "
            f"[{r['tail_type']}] "
            f"papers={r['paper_count']} "
            f"evidence={r['evidence_count']}"
        )

"""KG 候选召回与调试展示；最终生成上下文由 prompt_builder 构造。"""
from app.performance import timed

from collections import defaultdict
from app.kg_retriever import search_compound_knowledge
from app.evidence_retriever import get_evidence_by_edge_ids


@timed("kg_retrieval")
def retrieve_qa_candidates(
    compound,
    compound_node_id=None,
    compound_node_ids=None,
    relations=None,
    focus_terms=None,
    edge_limit=30,
    evidence_limit=50
):
    """
    检索、去重并均衡选择 KG 关系及关联证据，返回结构化候选。

    候选尚未经过最终相关性、主体归属筛选和重排，不能直接作为
    生成模型的支持材料。主流程筛选后由 reference_builder 分配引用，
    再由 prompt_builder 构造实际发送给模型的提示词。
    """

    # 1. 查 KG
    edges = search_compound_knowledge(
        compound=compound,
        compound_node_id=compound_node_id,
        compound_node_ids=compound_node_ids,
        relations=relations,
        focus_terms=focus_terms,
        limit=edge_limit
    )

    if not edges:
        return {
            "compound": compound,
            "edges": [],
            "evidence": []
        }

    # 2. 查 Evidence
    edge_ids = list(dict.fromkeys(
        source_edge_id
        for edge in edges
        for source_edge_id in (
            edge.get("source_edge_ids") or [edge["edge_id"]]
        )
        if source_edge_id
    ))

    # 先多取一些候选 evidence，
    # 避免直接 LIMIT 导致前几个 edge 把名额全部占满
    candidate_limit = max(
        evidence_limit * 5,
        200
    )

    all_evidence = get_evidence_by_edge_ids(
        edge_ids,
        limit=candidate_limit,
        focus_terms=focus_terms
    )

    # --------------------------------------------------------
    # Evidence 去重
    # 同一个 edge 下：
    # 同一篇论文 + 同一句 evidence 只保留一次
    #
    # 注意：
    # 不做跨 edge 全局去重，因为同一句证据可能同时支持
    # 多个不同 KG relation。
    # --------------------------------------------------------

    deduped_evidence = []
    seen = set()

    for ev in all_evidence:

        key = (
            ev.get("edge_id"),
            ev.get("evidence_sentence")
        )

        if key in seen:
            continue

        seen.add(key)
        deduped_evidence.append(ev)

    all_evidence = deduped_evidence
    # --------------------------------------------------------
    # 按 edge_id 分组
    # --------------------------------------------------------

    evidence_by_edge = defaultdict(list)

    for ev in all_evidence:
        evidence_by_edge[ev["edge_id"]].append(ev)

    # --------------------------------------------------------
    # 均衡选择 evidence
    #
    # 第一轮：每个 KG edge 尽量取 1 条
    # 第二轮开始：再给每个 relation 补第 2、3...条
    # 直到达到 evidence_limit
    # --------------------------------------------------------

    evidence = []

    round_index = 0

    while len(evidence) < evidence_limit:

        added = False

        for edge in edges:
            source_edge_ids = (
                edge.get("source_edge_ids")
                or [edge["edge_id"]]
            )
            candidates = [
                item
                for source_edge_id in source_edge_ids
                for item in evidence_by_edge.get(source_edge_id, [])
            ]

            if len(candidates) > round_index:

                evidence.append(
                    candidates[round_index]
                )

                added = True

            if len(evidence) >= evidence_limit:
                break

        # 所有 edge 都已经没有更多 evidence 了
        if not added:
            break

        round_index += 1

    return {"compound": compound, "edges": edges, "evidence": evidence}


def format_candidate_debug_text(candidates):
    """仅供人工查看原始候选；K/E 编号不是最终回答的 R 引用编号。"""
    compound = candidates["compound"]
    edges = candidates["edges"]
    evidence = candidates["evidence"]
    if not edges and not evidence:
        return ""

    # --------------------------------------------------------
    # 建立 edge_id -> edge 信息
    # --------------------------------------------------------

    edge_map = {
        source_edge_id: edge
        for edge in edges
        for source_edge_id in (
            edge.get("source_edge_ids") or [edge.get("edge_id")]
        )
        if source_edge_id
    }

    # --------------------------------------------------------
    # 构建候选调试文本（不发送给回答生成模型）
    # --------------------------------------------------------

    lines = []

    lines.append(
        f"Compound: {compound}"
    )

    lines.append("")
    lines.append("=== Knowledge Graph Relations ===")

    for i, e in enumerate(edges, 1):

        lines.append(
            f"[K{i}] "
            f"{e['head_name']} "
            f"-- {e['relation']} --> "
            f"{e['tail_name']} "
            f"[{e['tail_type']}] "
            f"(papers={e['paper_count']}, "
            f"evidence={e['evidence_count']})"
        )

    lines.append("")
    lines.append("=== Supporting Evidence ===")

    # --------------------------------------------------------
    # Evidence 编号只用于标识具体证据，
    # 后续最终参考文献编号应按 DOI / PMID 去重后单独生成
    # --------------------------------------------------------

    for i, ev in enumerate(evidence, 1):

        edge = edge_map.get(
            ev["edge_id"],
            {}
        )

        relation = edge.get(
            "relation",
            "unknown"
        )

        tail = edge.get(
            "tail_name",
            "unknown"
        )

        lines.append("")
        lines.append(f"[E{i}]")

        lines.append(
            f"KG relation: "
            f"{compound} -- {relation} --> {tail}"
        )

        lines.append(
            f"Title: {ev.get('title') or 'N/A'}"
        )

        lines.append(
            f"DOI: {ev.get('doi') or 'N/A'}"
        )

        lines.append(
            f"PMID: {ev.get('pmid') or 'N/A'}"
        )

        lines.append(
            f"Section: {ev.get('section') or 'N/A'}"
        )

        lines.append(
            f"Evidence: "
            f"{ev.get('evidence_sentence') or 'N/A'}"
        )

    return "\n".join(lines)


def build_qa_context(
    compound,
    compound_node_id=None,
    compound_node_ids=None,
    relations=None,
    focus_terms=None,
    edge_limit=30,
    evidence_limit=50
):
    """兼容旧脚本：context_text 仅是候选调试预览，不是最终 LLM 上下文。

    在线检索请使用 retrieve_qa_candidates，避免生成未使用的调试文本。
    """
    candidates = retrieve_qa_candidates(
        compound=compound,
        compound_node_id=compound_node_id,
        compound_node_ids=compound_node_ids,
        relations=relations,
        focus_terms=focus_terms,
        edge_limit=edge_limit,
        evidence_limit=evidence_limit
    )
    return {**candidates, "context_text": format_candidate_debug_text(candidates)}


if __name__ == "__main__":

    relations = [
        "inhibits",
        "activates",
        "regulates",
        "targets",
        "binds_to",
        "has_antitumor_activity"
    ]

    result = retrieve_qa_candidates(
        compound="Curcumin",
        relations=relations,
        edge_limit=15,
        evidence_limit=30
    )

    print(format_candidate_debug_text(result))

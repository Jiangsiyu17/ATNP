def filter_evidence_by_keywords(
    evidence,
    keywords=None,
    limit=20
):
    """
    根据问题关键词对 evidence 做简单相关性排序。

    流程：
    1. 关键词打分
    2. 按分数排序
    3. 全局去重：同一篇论文 + 同一句 evidence 只保留一次
    4. 返回 Top-N

    后续可以替换成 embedding / reranker / LLM relevance judge。
    """

    if not evidence:
        return []

    if not keywords:
        keywords = []

    keywords = [
        k.lower().strip()
        for k in keywords
        if k.strip()
    ]

    scored = []

    for ev in evidence:

        title = str(
            ev.get("title") or ""
        ).lower()

        sentence = str(
            ev.get("evidence_sentence") or ""
        ).lower()

        section = str(
            ev.get("section") or ""
        ).lower()

        text = " ".join([
            title,
            section,
            sentence
        ])

        score = 0

        for keyword in keywords:

            # evidence sentence 权重最高
            if keyword in sentence:
                score += 3

            # title 次之
            if keyword in title:
                score += 2

            # section 一般影响很小
            if keyword in section:
                score += 1

        scored.append(
            (score, ev)
        )

    # --------------------------------------------------------
    # 分数高的排前面
    # --------------------------------------------------------

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # 如果存在关键词命中的结果，
    # 优先只使用 score > 0
    # --------------------------------------------------------

    relevant = [
        (score, ev)
        for score, ev in scored
        if score > 0
    ]

    if not relevant:
        relevant = scored

    # --------------------------------------------------------
    # 最终给 LLM 前做全局去重
    #
    # 同一篇论文 + 同一句 evidence
    # 只保留一次。
    # --------------------------------------------------------

    final = []
    seen = set()

    for score, ev in relevant:

        key = (
            ev.get("paper_id"),
            ev.get("evidence_sentence")
        )

        if key in seen:
            continue

        seen.add(key)

        # 可选：把相关性分数带出去，
        # 后面调试非常有用
        ev_copy = dict(ev)
        ev_copy["relevance_score"] = score

        final.append(ev_copy)

        if len(final) >= limit:
            break

    return final
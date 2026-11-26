from app.performance import profile_qa, timed_call, record_progress
from app.retrieval_diagnostics import describe_retrieval
import logging
import re
import time
import unicodedata

from app.context_builder import retrieve_qa_candidates
from app.dynamic_reranker import expand_query, rerank_evidence
from app.evidence_compactor import (
    compact_intent_evidence,
    is_direct_intent_evidence
)
from app.prompt_builder import (
    INTENT_SECTION_LABELS,
    build_answer_repair_prompt,
    build_qa_prompt,
    intent_section_label,
    validate_answer_coverage
)
from app.search_agent import run_search_agent
from app.literature_tools import search_passages
from app.compound_discovery import parse_discovery_question, run_discovery
from app.multihop import parse_multihop_question, run_multihop
from app.source_answer import build_source_answer
from app.task_router import route_question, apply_routing, direct_response
from app.clinical_assessment import is_clinical_extrapolation, run_clinical_assessment
from app.model_client import call_llm
from app.question_parser import parse_question
from app.entity_resolver import resolve_compound_name
from app.reference_builder import build_references
from app.request_queue import check_cancelled
from app.citation_validator import validate_citations
from app.claim_validator import (
    build_supported_claims_answer,
    validate_answer_claims
)
from app.language_validator import (
    detect_question_language,
    validate_answer_language
)


LOGGER = logging.getLogger("atnp.qa_pipeline")
AUTO_CLAIM_VALIDATION_INTENTS = {
    "compound_identity",
    "anticancer_activity",
    "anticancer_mechanism",
    "pharmacological_effect",
    "target_mechanism"
}


PHARMACOLOGICAL_ANSWER_PATTERN = re.compile(
    r"(?:药理|治疗|疗效|抗炎|抗氧化|抗菌|抗微生物|抗癌|抗肿瘤|"
    r"免疫调节|细胞毒|抑制|激活|促进|降低|"
    r"pharmacolog|therapeutic|anti[- ]?inflamm|antioxid|"
    r"antimicrobial|antibacterial|anticancer|antitumou?r|cytotoxic|"
    r"\binhibit(?:s|ed|ion)?\b|\bactivat(?:e|es|ed|ion)\b)",
    re.IGNORECASE
)


def _answer_contains_pharmacological_claim(answer):
    return bool(PHARMACOLOGICAL_ANSWER_PATTERN.search(str(answer or "")))


def _preferred_compound_display_name(parsed, compound, response_language):
    if not str(response_language or "").startswith("zh"):
        return compound

    for suggestion in parsed.get("compound_suggestions") or []:
        if suggestion.get("canonical_name") != compound:
            continue
        alias = str(suggestion.get("suggested_alias") or "").strip()
        if alias:
            return f"{alias}（{compound}）"

    resolution = parsed.get("compound_resolution") or {}
    if resolution.get("canonical_name") == compound:
        matched = str(resolution.get("matched_term") or "").strip()
        if re.search(r"[\u3400-\u9fff]", matched):
            return f"{matched}（{compound}）"
    return compound


def _normalize_answer_compound_name(
    answer,
    parsed,
    compound,
    preferred_display_name
):
    text = str(answer or "")
    unresolved = str(
        parsed.get("unresolved_compound_mention") or ""
    ).strip()
    preferred_alias = str(preferred_display_name or "").split("（", 1)[0]
    if unresolved and preferred_alias and unresolved != preferred_alias:
        text = text.replace(unresolved, preferred_alias)
    return text


def _merge_grouped_items(group_results, item_key):
    """合并各意图的检索结果，并保留证据所属意图。"""

    merged = []
    index = {}

    for intent, items in group_results:
        for item in items:
            key = item_key(item)
            if key not in index:
                copied = dict(item)
                copied["intent_groups"] = [intent]
                index[key] = copied
                merged.append(copied)
                continue

            existing = index[key]
            existing["intent_groups"] = list(dict.fromkeys([
                *existing.get("intent_groups", []),
                intent
            ]))

    return merged


AGENT_BASELINE_MIN_STRONG_EVIDENCE = {
    "compound_identity": 1,
    "plant_source": 1,
    "toxicity": 1,
    "pharmacokinetics": 1,
    "anticancer_activity": 2,
    "anticancer_mechanism": 2,
    "pharmacological_effect": 2,
    "disease_association": 2,
    "target_mechanism": 2,
}


_CROSS_INTENT_QUERY_PATTERNS = {
    "plant_source": re.compile(
        r"\b(?:functions?|pharmacological effects?|activities|effects?)\b|"
        r"(?:功能|药理作用|生物活性|有哪些作用|有什么作用)",
        re.IGNORECASE
    ),
    "pharmacological_effect": re.compile(
        r"\b(?:plant sources?|botanical sources?|sources?|isolated from|"
        r"extracted from|derived from)\b|"
        r"(?:植物来源|来源于哪些植物|来自哪些植物|分离自|提取自)",
        re.IGNORECASE
    )
}


def _intent_retrieval_question(question, intent, requested_intents):
    """Remove words belonging only to a sibling sub-question.

    Without this split, an English question containing ``plant source`` and
    ``functions`` made the effect reranker focus on the words ``plant`` and
    ``source``.  Methods and constituent sentences then outranked actual
    pharmacological outcomes.
    """

    if len(requested_intents or []) < 2:
        return question
    pattern = _CROSS_INTENT_QUERY_PATTERNS.get(intent)
    if pattern is None:
        return question
    cleaned = pattern.sub(" ", str(question or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,，、;；?？")
    return cleaned or question


def _assess_agent_need(intent, selected, stats, edges=None):
    """Decide whether deterministic retrieval needs adaptive supplementation.

    Reranking deliberately fills a minimum context even when rows fall below
    the relevance threshold.  Agent gating must therefore count only rows at
    or above that threshold, rather than treating every filled context item as
    strong evidence.
    """

    required = AGENT_BASELINE_MIN_STRONG_EVIDENCE.get(intent, 1)
    threshold = float(stats.get("score_threshold") or 0.0)
    edge_by_id = {
        source_id: edge
        for edge in (edges or [])
        for source_id in (
            edge.get("source_edge_ids") or [edge.get("edge_id")]
        )
        if source_id
    }
    strong = [
        row for row in (selected or [])
        if row.get("evidence_sentence")
        and float(row.get("rerank_score") or 0.0) >= threshold
        and is_direct_intent_evidence(
            intent,
            row,
            edge_by_id.get(row.get("edge_id"), {})
        )
    ]
    sufficient = len(strong) >= required
    return {
        "sufficient": sufficient,
        "required_strong_evidence": required,
        "strong_evidence_count": len(strong),
        "selected_evidence_count": len(selected or []),
        "score_threshold": threshold,
        "strong_evidence_ids": [row.get("evidence_id") for row in strong],
        "reason": (
            "baseline_strong_evidence_sufficient"
            if sufficient else
            "baseline_strong_evidence_below_requirement"
        )
    }


def _skipped_agent_metadata(assessment):
    return {
        "triggered": False,
        "trigger_reason": assessment["reason"],
        "stop_reason": "baseline_sufficient",
        "stop_message": "默认知识图谱与本地全文检索已达到证据覆盖要求，未启动 Search Agent",
        "trace": [],
        "tool_call_count": 0,
        "decision_count": 0,
        "recovery_count": 0,
        "baseline_fallback_used": False,
        "elapsed_ms": 0,
        "baseline_assessment": assessment,
        "retrieval_scope": "local_only"
    }


def _replace_markdown_section(answer, replacement, intent):
    """Replace one generated section with deterministic grounded content."""

    if not replacement:
        return answer
    labels = {
        str(label).casefold()
        for label in INTENT_SECTION_LABELS[intent].values()
    }
    lines = str(answer or "").splitlines()
    kept = []
    skipping = False
    for line in lines:
        heading = re.sub(
            r"^[#\s]+", "", line
        ).strip().rstrip(":：").casefold()
        is_heading = bool(re.match(r"^\s*#{1,6}\s+", line))
        if is_heading and heading in labels:
            skipping = True
            continue
        if skipping and is_heading:
            skipping = False
        if not skipping:
            kept.append(line)
    remainder = "\n".join(kept).strip()
    return replacement.strip() + (
        "\n\n" + remainder if remainder else ""
    )


@profile_qa
def run_qa(
    question,
    compound=None,
    context_compound=None,
    model=None,
    relations=None,
    keywords=None,
    edge_limit=100,
    evidence_limit=300,
    final_evidence_limit=None,
    rerank_min_evidence=6,
    rerank_max_evidence=25,
    rerank_token_budget=7000,
    rerank_score_threshold=0.45,
    call_model=False,
    model_max_tokens=1500,
    model_timeout=180,
    validate_claims=None,
    claim_validation_max_tokens=2000,
    claim_validation_timeout=600,
    progress_callback=None,
    search_agent=False,
    task_router=False,
    external_search=False,
    rerank_backend=None
):
    """
    ATNP Expert Q&A 主流程。

    当前流程：
    1. 问题解析与化合物实体识别
    2. 查询扩展
    3. KG 与候选证据扩展召回
    4. 动态证据重排与 token 预算筛选
    5. Relevant edge filtering
    6. 结构化参考文献构建
    7. Strict RAG prompt
    8. 可选：调用 LLM 并校验引用
    9. 可选：逐条判定结论是否被引用证据支持
    """

    if external_search:
        raise ValueError('External search has been removed; only local database and fulltext retrieval are supported.')
    started_at = time.perf_counter()

    def emit(stage, state, message, **details):
        check_cancelled()
        record_progress(stage, state, details)
        if progress_callback is None:
            return

        event = {
            "stage": stage,
            "state": state,
            "message": message,
            "elapsed_ms": round(
                (time.perf_counter() - started_at) * 1000
            ),
            "details": details
        }

        try:
            progress_callback(event)
        except Exception:
            # Progress reporting must never interrupt the QA result.
            LOGGER.exception("QA progress callback failed")

    # A two-hop discovery must run before compound parsing/context inheritance.
    multihop = timed_call("multihop_parse", parse_multihop_question,
        question, judge=(lambda prompt: timed_call("multihop_planning", call_llm,
            prompt, model=model, temperature=0.0, max_tokens=512, timeout=min(model_timeout, 90)
        )) if call_model else None, event=emit
    ) if compound is None else None
    if multihop is not None:
        verify = call_model and validate_claims is not False
        return timed_call("multihop", run_multihop,
            question, multihop, model=model, evidence_limit=evidence_limit,
            activity_judge=(lambda prompt: timed_call("multihop_activity_validation", call_llm,
                prompt, model=model, temperature=0.0, max_tokens=512, timeout=model_timeout
            )) if verify else None,
            source_judge=(lambda prompt: timed_call("multihop_source_validation", call_llm,
                prompt, model=model, temperature=0.0, max_tokens=1000, timeout=min(model_timeout, 90)
            )) if verify else None,
            relation_judge=(lambda prompt: timed_call("multihop_relation_validation", call_llm,
                prompt, model=model, temperature=0.0, max_tokens=1000, timeout=min(model_timeout, 90)
            )) if verify else None,
            event=emit, rerank_backend=rerank_backend)

    # Discovery has no root compound and must run before compound parsing,
    # model routing, and inheritance of a previous conversation's compound.
    discovery = timed_call("discovery_parse", parse_discovery_question,
        question, judge=(lambda prompt: timed_call("discovery_planning", call_llm,
            prompt, model=model, temperature=0.0, max_tokens=512, timeout=min(model_timeout, 90)
        )) if call_model else None, event=emit
    ) if compound is None else None
    if discovery is not None:
        return timed_call("discovery", run_discovery,
            question, discovery, model=model, evidence_limit=evidence_limit,
            judge=(lambda prompt: timed_call("discovery_activity_validation", call_llm,
                prompt, model=model, temperature=0.0,
                max_tokens=512, timeout=model_timeout
            )) if call_model and validate_claims is not False else None,
            event=emit, rerank_backend=rerank_backend
        )

    emit(
        "question_parsing",
        "running",
        "正在识别天然产物和问题意图"
    )
    relations_overridden = relations is not None
    keywords_overridden = keywords is not None
    parsed = timed_call("question_parsing", parse_question,question)
    response_language = detect_question_language(question)
    clinical_task = is_clinical_extrapolation(question)
    if task_router and call_model and not clinical_task:
        emit("task_routing", "running", "正在判断问题领域、检索需求和问题意图")
        route = route_question(
            question, compound or context_compound,
            judge=lambda prompt: timed_call("task_routing", call_llm,prompt, model=model, temperature=0.0,
                max_tokens=512, timeout=min(model_timeout, 90))
        )
        # Existing verified/ambiguous chemical mentions cannot be bypassed by a direct route.
        if compound or parsed.get("compound") or parsed.get("ambiguous_compound"):
            if not route['requires_retrieval']:
                route['guard_reason'] = 'chemical_mention_requires_evidence'
            route['requires_retrieval'] = True
        # Keep direct responses limited to fixed non-scientific templates.
        if route['domain'] == 'uncertain':
            route['requires_retrieval'] = True
        parsed = apply_routing(parsed, route)
        if route['status'] == 'ok' and route.get('entity_mention') and not (
            parsed.get('compound') or parsed.get('ambiguous_compound')
            or parsed.get('unresolved_compound_mention')
        ):
            entity = timed_call("question_parsing", parse_question,route['entity_mention'] + ' 有什么作用？')
            for key in ('compound', 'compound_node_id', 'compound_node_ids',
                        'compound_resolution', 'ambiguous_compound',
                        'unresolved_compound_mention', 'compound_suggestions'):
                parsed[key] = entity.get(key)
        emit("task_routing", "completed",
             "问题路由完成，进入证据检索" if route['requires_retrieval'] else "问题路由完成，无需检索专业证据",
             domain=route['domain'], requires_retrieval=route['requires_retrieval'],
             intents=parsed.get('intents'), routing_status=route['status'])
        if route['status'] == 'ok' and not route['requires_retrieval']:
            answer = direct_response(route, response_language)
            return {'status': 'out_of_scope' if route['task'] == 'out_of_scope' else 'ok',
                    'question': question, 'model': model, 'compound': None, 'parsed': parsed,
                    'answer': answer, 'message': None, 'references': [],
                    'candidate_edges': [], 'candidate_evidence': [], 'relevant_edges': [],
                    'relevant_evidence': [], 'prompt': None,
                    'retrieval': {'mode': 'direct', 'task_routing': route, 'reference_count': 0},
                    'citation_validation': None, 'claim_validation': None,
                    'language_validation': validate_answer_language(answer, response_language)}
    parsed_intents = list(dict.fromkeys(
        parsed.get("intents") or [parsed.get("intent")]
    ))
    parsed_intents = [intent for intent in parsed_intents if intent]
    effective_validate_claims = (
        bool(
            set(parsed_intents).intersection(
                AUTO_CLAIM_VALIDATION_INTENTS
            )
        )
        if validate_claims is None
        else bool(validate_claims)
    )
    claim_validation_mode = (
        "auto"
        if validate_claims is None
        else ("always" if validate_claims else "disabled")
    )

    explicit_compound = parsed["compound"]
    compound_node_id = parsed.get("compound_node_id")
    compound_node_ids = list(parsed.get("compound_node_ids") or [])
    ambiguous_compound = parsed.get("ambiguous_compound")
    unresolved_compound_mention = parsed.get(
        "unresolved_compound_mention"
    )
    compound_suggestions = list(
        parsed.get("compound_suggestions") or []
    )
    compound_inherited = False
    confirmed_override = False

    if compound is None:
        compound = explicit_compound

        can_inherit_context = (
            compound is None
            and ambiguous_compound is None
            and unresolved_compound_mention is None
        )

        if can_inherit_context and context_compound is not None:
            context_name = str(context_compound).strip()
            context_resolution = resolve_compound_name(context_name)

            if context_resolution["status"] == "resolved":
                compound = context_resolution["canonical_name"]
                compound_node_id = context_resolution["compound_node_id"]
                compound_node_ids = context_resolution["compound_node_ids"]
                compound_inherited = True
    else:
        supplied_name = str(compound).strip()
        supplied_resolution = resolve_compound_name(supplied_name)

        if supplied_resolution["status"] == "resolved":
            confirmed_override = True
            compound = supplied_resolution["canonical_name"]
            compound_node_id = supplied_resolution["compound_node_id"]
            compound_node_ids = supplied_resolution["compound_node_ids"]
        elif supplied_resolution["status"] == "ambiguous":
            compound = None
            ambiguous_compound = {
                "mention": supplied_resolution["mention"],
                "candidates": [
                    item["canonical_name"]
                    for item in supplied_resolution["candidates"]
                ]
            }
        else:
            compound = None
            unresolved_compound_mention = supplied_name

    # Compatibility for tests and internal callers that provide only the
    # canonical string in a mocked parse result.
    if compound and not compound_node_ids:
        resolution = resolve_compound_name(compound)
        if resolution["status"] == "resolved":
            compound = resolution["canonical_name"]
            compound_node_id = resolution["compound_node_id"]
            compound_node_ids = resolution["compound_node_ids"]

    retrieval_question = question
    if confirmed_override:
        original_mention = (
            unresolved_compound_mention
            or (parsed.get("compound_resolution") or {}).get("mention")
        )
        if original_mention and " / " not in original_mention:
            retrieval_question, replacement_count = re.subn(
                re.escape(original_mention), lambda _: compound,
                question, flags=re.IGNORECASE
            )
            if not replacement_count:
                normalized_question = unicodedata.normalize("NFKC", question)
                normalized_mention = unicodedata.normalize("NFKC", original_mention)
                retrieval_question = re.sub(
                    re.escape(normalized_mention), lambda _: compound,
                    normalized_question, flags=re.IGNORECASE
                )

    parsed = dict(parsed)
    parsed["original_question"] = question
    parsed["retrieval_question"] = retrieval_question
    parsed["confirmed_name_applied"] = retrieval_question != question
    parsed["claim_validation_mode"] = claim_validation_mode
    parsed["claim_validation_enabled"] = effective_validate_claims
    parsed["intents"] = parsed_intents
    parsed["question_language"] = response_language
    parsed["response_language"] = response_language
    parsed["explicit_compound"] = explicit_compound
    parsed["compound"] = compound
    parsed["compound_node_id"] = compound_node_id
    parsed["compound_node_ids"] = compound_node_ids
    parsed["compound_inherited"] = compound_inherited
    parsed["context_inheritance_blocked"] = bool(
        compound is None
        and (
            ambiguous_compound is not None
            or unresolved_compound_mention is not None
        )
    )

    if relations is None:
        relations = parsed["relations"]
    else:
        relations = list(relations)

    if keywords is None:
        keywords = parsed["keywords"]
    else:
        keywords = list(keywords)

    raw_intent_queries = parsed.get("intent_queries") or [{
        "intent": parsed.get("intent"),
        "relations": relations,
        "keywords": keywords
    }]
    intent_queries = []
    for group in raw_intent_queries:
        group_intent = group.get("intent") or parsed.get("intent")
        group_relations = (
            relations
            if relations_overridden
            else list(group.get("relations") or [])
        )
        group_keywords = (
            keywords
            if keywords_overridden
            else list(group.get("keywords") or [])
        )
        intent_queries.append({
            "intent": group_intent,
            "relations": list(dict.fromkeys(group_relations)),
            "keywords": list(dict.fromkeys(group_keywords))
        })
    parsed["intent_queries"] = intent_queries

    if ambiguous_compound is not None and compound is None:
        parsed_message = (
            "检测到有歧义的化合物名称："
            f"{ambiguous_compound['mention']}"
        )
    elif unresolved_compound_mention is not None and compound is None:
        parsed_message = (
            "检测到未匹配的新化合物名称："
            f"{unresolved_compound_mention}"
        )
    elif compound_inherited:
        parsed_message = f"已沿用上文天然产物：{compound}"
    elif compound:
        parsed_message = f"已识别天然产物：{compound}"
    else:
        parsed_message = "未识别到明确的天然产物"
    emit(
        "question_parsing",
        "completed",
        parsed_message,
        compound=compound,
        compound_node_id=compound_node_id,
        compound_node_ids=compound_node_ids,
        compound_inherited=compound_inherited,
        context_inheritance_blocked=parsed[
            "context_inheritance_blocked"
        ],
        ambiguous_compound=ambiguous_compound,
        unresolved_compound_mention=unresolved_compound_mention,
        compound_suggestions=compound_suggestions,
        intent=parsed.get("intent"),
        intents=parsed_intents,
        relations=relations,
        keywords=keywords
    )

    if compound is None:
        if ambiguous_compound is not None:
            candidates = ambiguous_compound["candidates"]
            clarification_message = (
                f"“{ambiguous_compound['mention']}”可能指不同化合物。"
                f"请选择明确的标准名称：{'、'.join(candidates)}。"
                "本次未沿用上文中的化合物。"
            )
        elif unresolved_compound_mention is not None:
            suggestion_text = ""
            if compound_suggestions:
                choices = "、".join(
                    f"{item['canonical_name']}"
                    f"（{item['suggested_alias']}）"
                    for item in compound_suggestions[:3]
                )
                suggestion_text = f"您是否指 {choices}？"
                if any(item.get("requires_suffix_confirmation") for item in compound_suggestions):
                    suggestion_text += "名称后缀尚未核实，候选不代表与原名称相同。"
                if len(compound_suggestions) == 1:
                    suggestion_text += "若确实指该候选，请回复“是的”，我会按候选重新检索原问题。"

            clarification_message = (
                f"检测到可能的新化合物名称“{unresolved_compound_mention}”，"
                "但尚未匹配到 ATNP 标准名称或中文别名。"
                "本次未沿用上文中的化合物；请使用数据库英文标准名，"
                f"或选择确认候选。{suggestion_text}"
            )
        else:
            clarification_message = (
                "未识别出需要查询的化合物，"
                "请提供化合物的中文或英文名称。"
            )

        return {
            "status": "needs_clarification",
            "message": clarification_message,
            "question": question,
            "compound": None,
            "parsed": parsed,
            "candidate_edges": [],
            "candidate_evidence": [],
            "relevant_edges": [],
            "relevant_evidence": [],
            "references": [],
            "prompt": None,
            "answer": None,
            "citation_validation": None,
            "language_validation": None,
            "claim_validation": None,
            "retrieval": {
                "expanded_terms": [],
                "candidate_edge_count": 0,
                "candidate_evidence_count": 0,
                "selected_evidence_count": 0
            }
        }

    if clinical_task:
        return timed_call("clinical_assessment", run_clinical_assessment,question, parsed, model=model,
            judge=(lambda prompt: timed_call("clinical_validation", call_llm,prompt, model=model, temperature=0.0,
                max_tokens=512, timeout=min(model_timeout, 45)))
                if call_model and validate_claims is not False else None, event=emit)

    expanded_groups = []
    expanded_term_map = {}
    specific_query_terms = []

    for group in intent_queries:
        group_retrieval_question = _intent_retrieval_question(
            retrieval_question,
            group["intent"],
            parsed_intents
        )
        expanded = expand_query(
            question=group_retrieval_question,
            compound=compound,
            intent=group["intent"],
            keywords=group["keywords"],
            relations=group["relations"]
        )
        expanded_groups.append({
            **group,
            "expanded": expanded,
            "retrieval_question": group_retrieval_question
        })
        for item in expanded["terms"]:
            existing = expanded_term_map.get(item["term"])
            if existing is None or item["weight"] > existing["weight"]:
                expanded_term_map[item["term"]] = item
        specific_query_terms.extend(expanded["specific_terms"])

    expanded_terms = sorted(
        expanded_term_map.values(),
        key=lambda item: (-item["weight"], item["term"])
    )
    parsed["expanded_query_terms"] = [
        item["term"] for item in expanded_terms
    ]
    parsed["specific_query_terms"] = list(dict.fromkeys(
        specific_query_terms
    ))
    emit(
        "query_expansion",
        "completed",
        f"已按 {len(expanded_groups)} 个意图扩展查询",
        intents=parsed_intents,
        retrieval_question=retrieval_question,
        expanded_terms=expanded_terms,
        specific_terms=parsed["specific_query_terms"]
    )

    # --------------------------------------------------------
    # 1. 构建候选 KG + Evidence
    # --------------------------------------------------------

    emit(
        "knowledge_retrieval",
        "running",
        "正在检索相关知识图谱关系"
    )
    group_count = len(expanded_groups)

    def allocate(total, index):
        total = max(group_count, int(total))
        return total // group_count + (1 if index < total % group_count else 0)

    total_rerank_max = max(group_count, int(rerank_max_evidence))
    if final_evidence_limit is not None:
        total_rerank_max = max(
            group_count,
            min(total_rerank_max, int(final_evidence_limit))
        )

    candidate_edge_groups = []
    candidate_evidence_groups = []
    selected_evidence_groups = []
    rerank_group_stats = []

    for index, group in enumerate(expanded_groups):
        retrieval_focus_terms = list(
            group["expanded"]["specific_terms"]
        )
        if group["intent"] == "compound_identity":
            retrieval_focus_terms.extend([
                "prodrug",
                "pro-drug",
                "selective inhibitor",
                "coxib",
                "drug class"
            ])
        def kg_search(agent_keywords):
            return retrieve_qa_candidates(
                compound=compound,
                compound_node_id=compound_node_id,
                compound_node_ids=compound_node_ids,
                relations=group["relations"],
                focus_terms=list(dict.fromkeys(retrieval_focus_terms + agent_keywords)),
                edge_limit=allocate(edge_limit, index),
                evidence_limit=allocate(evidence_limit, index)
            )

        group_max = allocate(total_rerank_max, index)
        group_min = min(int(rerank_min_evidence), group_max)
        def evaluate_candidates(evidence, edges):
            return rerank_evidence(
                question=group["retrieval_question"], evidence=evidence, edges=edges,
                compound=compound, intent=group['intent'], keywords=group['keywords'],
                relations=group['relations'], min_evidence=max(1,group_min),
                max_evidence=max(1,group_max), token_budget=allocate(rerank_token_budget,index),
                score_threshold=rerank_score_threshold, rerank_backend=rerank_backend)

        # Always perform the cheap, deterministic baseline first.  Search
        # Agent is a recovery controller for evidence gaps, not the default
        # entry point for every well-formed query.
        baseline_result = kg_search([])
        baseline_edges = baseline_result["edges"]
        baseline_evidence = baseline_result["evidence"]
        emit("fulltext_retrieval", "running", "正在补充检索本地全文证据", intent=group["intent"])
        fulltext = search_passages(compound, retrieval_focus_terms or group["keywords"])
        baseline_evidence = list({
            row["evidence_id"]: row
            for row in [*baseline_evidence, *fulltext["evidence"]]
        }.values())
        emit("fulltext_retrieval", "completed", f"本地全文召回 {len(fulltext['evidence'])} 条候选证据",
             intent=group["intent"], source_status=fulltext["tool_status"], passage_count=fulltext.get("passage_count", 0))
        emit("relevance_reranking", "running", "正在动态重排候选证据", intent=group["intent"])
        baseline_selected, baseline_stats = evaluate_candidates(
            baseline_evidence,
            baseline_edges
        )
        baseline_assessment = _assess_agent_need(
            group["intent"],
            baseline_selected,
            baseline_stats,
            baseline_edges
        )

        agent_metadata = None
        if search_agent and call_model and not baseline_assessment["sufficient"]:
            emit(
                f"search_agent_{group['intent']}",
                "running",
                "默认检索证据覆盖不足，正在启动 Search Agent 补查",
                **baseline_assessment
            )
            baseline_keywords = retrieval_focus_terms or group["keywords"]
            initial_seen_actions = {
                ("search_kg", ()),
                ("search_passages", tuple(sorted(
                    keyword.casefold() for keyword in baseline_keywords
                )))
            }
            agent_result = timed_call(
                "search_agent",
                run_search_agent,
                question=retrieval_question,
                compound=compound,
                intent=group["intent"],
                judge=lambda prompt: timed_call(
                    "search_agent_decision",
                    call_llm,
                    prompt,
                    model=model,
                    temperature=0.0,
                    max_tokens=512,
                    timeout=min(model_timeout, 90)
                ),
                kg_search=kg_search,
                evaluator=evaluate_candidates,
                external_search=external_search,
                initial_evidence=baseline_selected,
                initial_edges=baseline_edges,
                initial_eligible_ids=baseline_assessment[
                    "strong_evidence_ids"
                ],
                initial_feedback={
                    "status": "baseline_coverage_insufficient",
                    "selected_count": len(baseline_selected),
                    "eligible_evidence_ids": baseline_assessment[
                        "strong_evidence_ids"
                    ],
                    **baseline_assessment
                },
                initial_seen_actions=initial_seen_actions,
                event=lambda state, message, **details: emit(
                    f"search_agent_{group['intent']}",
                    state,
                    message,
                    **details
                )
            )
            group_edges = list({
                row["edge_id"]: row
                for row in [*baseline_edges, *agent_result["edges"]]
                if row.get("edge_id")
            }.values())
            group_evidence = list({
                row["evidence_id"]: row
                for row in [*baseline_evidence, *agent_result["evidence"]]
                if row.get("evidence_id")
            }.values())
            selected, group_stats = evaluate_candidates(
                group_evidence,
                group_edges
            )
            agent_metadata = {
                **(agent_result.get("agent") or {}),
                "triggered": True,
                "trigger_reason": baseline_assessment["reason"],
                "baseline_assessment": baseline_assessment
            }
        else:
            group_edges = baseline_edges
            group_evidence = baseline_evidence
            selected = baseline_selected
            group_stats = baseline_stats
            if search_agent and call_model:
                agent_metadata = _skipped_agent_metadata(
                    baseline_assessment
                )
                emit(
                    f"search_agent_{group['intent']}",
                    "completed",
                    "默认检索证据已达到覆盖要求，未启动 Search Agent",
                    triggered=False,
                    **baseline_assessment
                )

        selected, compaction_stats = compact_intent_evidence(
            intent=group["intent"],
            evidence=selected,
            edges=group_edges
        )
        group_stats = {
            **group_stats,
            "selected_count": len(selected),
            "estimated_tokens": compaction_stats[
                "after_estimated_tokens"
            ],
            "evidence_compaction": compaction_stats
        }
        group_stats["fulltext_retrieval"] = {
            "status": fulltext["tool_status"], "passage_count": fulltext.get("passage_count", 0),
            "returned_count": len(fulltext["evidence"]), "error_type": fulltext.get("error_type"),
            "selected_count": sum(r.get("retrieval_source") == "local_passage" for r in selected)
        }
        candidate_edge_groups.append((group["intent"], group_edges))
        candidate_evidence_groups.append((group["intent"], group_evidence))
        selected_evidence_groups.append((group["intent"], selected))
        rerank_group_stats.append({
            "intent": group["intent"],
            "relations": group["relations"],
            "keywords": group["keywords"],
            "search_agent": agent_metadata,
            "agent_triggered": bool(
                agent_metadata and agent_metadata.get("triggered")
            ),
            "baseline_sufficiency": baseline_assessment,
            "candidate_edge_count": len(group_edges),
            "candidate_evidence_count": len(group_evidence),
            **group_stats
        })

    candidate_edges = _merge_grouped_items(
        candidate_edge_groups,
        lambda item: item.get("edge_id") or (
            item.get("relation"), item.get("tail_name")
        )
    )
    candidate_evidence = _merge_grouped_items(
        candidate_evidence_groups,
        lambda item: (
            item.get("evidence_id"),
            item.get("edge_id"),
            item.get("evidence_sentence")
        )
    )
    relevant_evidence = _merge_grouped_items(
        selected_evidence_groups,
        lambda item: (
            item.get("evidence_id"),
            item.get("edge_id"),
            item.get("evidence_sentence")
        )
    )

    emit(
        "knowledge_retrieval",
        "completed",
        f"按意图选取 {len(candidate_edges)} 条候选知识图谱关系",
        edge_count=len(candidate_edges),
        edge_limit=edge_limit,
        intent_groups=[
            {
                "intent": item["intent"],
                "edge_count": item["candidate_edge_count"]
            }
            for item in rerank_group_stats
        ]
    )
    emit(
        "evidence_retrieval",
        "completed",
        f"按意图选取 {len(candidate_evidence)} 条候选证据",
        evidence_count=len(candidate_evidence),
        evidence_limit=evidence_limit,
        intent_groups=[
            {
                "intent": item["intent"],
                "evidence_count": item["candidate_evidence_count"]
            }
            for item in rerank_group_stats
        ]
    )

    # --------------------------------------------------------
    # 2. Evidence 相关性筛选
    # --------------------------------------------------------

    rejection_reasons = {}
    for stats in rerank_group_stats:
        for reason, count in stats[
            "compound_attribution_rejection_reasons"
        ].items():
            rejection_reasons[reason] = (
                rejection_reasons.get(reason, 0) + count
            )

    numeric_thresholds = [
        item["effective_threshold"]
        for item in rerank_group_stats
        if item["effective_threshold"] is not None
    ]
    rerank_stats = {
        "candidate_count": len(candidate_evidence),
        "deduplicated_count": sum(
            item["deduplicated_count"] for item in rerank_group_stats
        ),
        "compound_attribution_filtered_count": sum(
            item["compound_attribution_filtered_count"]
            for item in rerank_group_stats
        ),
        "compound_attribution_rejection_reasons": rejection_reasons,
        "intent_filtered_count": sum(
            item["intent_filtered_count"] for item in rerank_group_stats
        ),
        "compaction_removed_count": sum(
            item["evidence_compaction"]["removed_count"]
            for item in rerank_group_stats
        ),
        "selected_count": len(relevant_evidence),
        "dynamic_target": sum(
            item["dynamic_target"] for item in rerank_group_stats
        ),
        "score_threshold": rerank_score_threshold,
        "base_score_threshold": rerank_score_threshold,
        "effective_threshold": (
            round(min(numeric_thresholds), 6)
            if numeric_thresholds else None
        ),
        "cutoff_reason": (
            "multi_intent"
            if group_count > 1
            else rerank_group_stats[0]["cutoff_reason"]
        ),
        "estimated_tokens": sum(
            item["estimated_tokens"] for item in rerank_group_stats
        ),
        "token_budget": rerank_token_budget,
        "expanded_terms": expanded_terms,
        "specific_terms": parsed["specific_query_terms"],
        "focused_candidate_count": sum(
            item["focused_candidate_count"] for item in rerank_group_stats
        ),
        "intent_groups": rerank_group_stats
    }
    emit(
        "relevance_reranking",
        "completed",
        f"分意图重排后保留 {len(relevant_evidence)} 条高相关证据",
        evidence_count=len(relevant_evidence),
        dynamic_target=rerank_stats["dynamic_target"],
        score_threshold=rerank_stats["score_threshold"],
        effective_threshold=rerank_stats["effective_threshold"],
        focused_candidate_count=rerank_stats["focused_candidate_count"],
        compound_attribution_filtered_count=rerank_stats[
            "compound_attribution_filtered_count"
        ],
        compound_attribution_rejection_reasons=rerank_stats[
            "compound_attribution_rejection_reasons"
        ],
        compaction_removed_count=rerank_stats[
            "compaction_removed_count"
        ],
        cutoff_reason=rerank_stats["cutoff_reason"],
        estimated_tokens=rerank_stats["estimated_tokens"],
        token_budget=rerank_stats["token_budget"],
        intent_groups=rerank_group_stats
    )

    # --------------------------------------------------------
    # 3. 根据最终 Evidence 反向筛选 KG edges
    # --------------------------------------------------------

    relevant_edge_ids = {
        ev["edge_id"]
        for ev in relevant_evidence
    }

    relevant_edges = [
        edge
        for edge in candidate_edges
        if relevant_edge_ids.intersection(
            edge.get("source_edge_ids") or [edge["edge_id"]]
        )
    ]

    # --------------------------------------------------------
    # 4. 按论文去重并构建结构化参考文献
    # --------------------------------------------------------

    references = build_references(
        evidence=relevant_evidence,
        edges=relevant_edges
    )
    emit(
        "reference_building",
        "completed",
        f"整理为 {len(references)} 篇可引用文献",
        reference_count=len(references)
    )

    retrieval_stats = {
        **rerank_stats,
        "candidate_edge_count": len(candidate_edges),
        "candidate_evidence_count": len(candidate_evidence),
        "reference_count": len(references),
        "retrieval_question": retrieval_question
    }

    retrieval_stats.update(describe_retrieval(rerank_group_stats, references))

    for group_stats in retrieval_stats["intent_groups"]:
        group_stats["reference_ids"] = [
            reference["id"]
            for reference in references
            if group_stats["intent"] in reference.get(
                "intent_groups", []
            )
        ]

    # --------------------------------------------------------
    # 5. 从最终 references 构建实际发送给模型的 Strict RAG Prompt。
    # 不拼接候选阶段的调试文本，避免重新引入已淘汰证据及 K/E 编号。
    # --------------------------------------------------------

    preferred_compound_name = _preferred_compound_display_name(
        parsed,
        compound,
        response_language
    )
    parsed["preferred_compound_name"] = preferred_compound_name
    prompt = build_qa_prompt(
        question=retrieval_question,
        compound=compound,
        edges=relevant_edges,
        references=references,
        response_language=response_language,
        requested_intents=parsed_intents,
        preferred_compound_name=preferred_compound_name
    )

    # --------------------------------------------------------
    # 6. 是否真正调用模型
    # --------------------------------------------------------

    answer = None
    citation_validation = None
    language_validation = None
    claim_validation = None
    answer_coverage_validation = None
    generation_attempts = 0
    repair_reason = None
    safe_fallback_applied = False
    coverage_fallback_applied = False

    if not references:
        status = "no_evidence"
        message = retrieval_stats["evidence_message"]
    else:
        status = "ready"
        message = None

    if call_model and references:
        emit(
            "answer_generation",
            "running",
            "正在根据检索证据生成回答",
            reference_count=len(references),
            evidence_count=len(relevant_evidence)
        )
        verified_source_answer = (
            build_source_answer(references, response_language)
            if "plant_source" in parsed_intents else None
        )
        source_answer = (
            verified_source_answer
            if parsed_intents == ["plant_source"] else None
        )
        if source_answer:
            retrieval_stats["answer_mode"] = "verbatim_plant_sources"

        def generate(generation_prompt):
            nonlocal generation_attempts
            if source_answer:
                return source_answer
            generation_attempts += 1
            return timed_call(("answer_generation" if generation_attempts == 1 else "answer_repair"), call_llm,
                prompt=generation_prompt,
                model=model,
                temperature=0.0,
                max_tokens=model_max_tokens,
                timeout=model_timeout
            )

        def run_claim_validation(
            current_answer,
            previous_validation=None
        ):
            emit(
                "claim_validation",
                "running",
                "正在核验回答中的声明"
            )
            validation = timed_call("claim_validation_total", validate_answer_claims,
                answer=current_answer,
                references=references,
                compound=compound,
                previous_validation=previous_validation,
                judge=lambda judge_prompt: timed_call("claim_validation", call_llm,
                    prompt=judge_prompt,
                    model=model,
                    temperature=0.0,
                    max_tokens=claim_validation_max_tokens,
                    timeout=claim_validation_timeout
                )
            )

            emit(
                "claim_validation",
                "completed",
                "声明级证据核验完成",
                valid=validation["valid"],
                total_claims=validation["total_claims"],
                supported_claims=validation["supported_claims"],
                reused_claims=validation.get("reused_claims", 0),
                judged_claims=validation.get("judged_claims", 0)
            )
            return validation

        def validate_generated_answer(raw_text):
            validation = validate_citations(raw_text, references)
            validation["answer"] = _normalize_answer_compound_name(
                validation["answer"],
                parsed,
                compound,
                preferred_compound_name
            )
            return validation

        raw_answer = generate(prompt)
        citation_validation = validate_generated_answer(raw_answer)
        answer = citation_validation["answer"]
        language_validation = validate_answer_language(
            answer,
            response_language
        )
        answer_coverage_validation = validate_answer_coverage(
            answer,
            parsed_intents,
            response_language
        )

        if (
            not citation_validation["valid"]
            or not language_validation["valid"]
            or not answer_coverage_validation["valid"]
        ):
            repair_failures = []
            if not citation_validation["valid"]:
                repair_failures.append(
                    "Citation validation failed. Invalid identifiers: "
                    f"{citation_validation['invalid_reference_ids']}; "
                    "missing citations: "
                    f"{citation_validation['missing_citations']}."
                )
            if not language_validation["valid"]:
                repair_failures.append(
                    "Output-language validation failed. Expected "
                    f"{response_language}, but detected "
                    f"{language_validation['detected_language']}."
                )
            if not answer_coverage_validation["valid"]:
                repair_failures.append(
                    "Mandatory multi-intent sections are missing: "
                    f"{answer_coverage_validation['missing_intents']}."
                )
            repair_reason = " ".join(repair_failures)
            emit(
                "answer_repair",
                "running",
                "回答格式或语言未通过校验，正在自动修复",
                reason="answer_validation",
                missing_intents=answer_coverage_validation[
                    "missing_intents"
                ]
            )
            repair_prompt = build_answer_repair_prompt(
                base_prompt=prompt,
                previous_answer=raw_answer,
                reason=repair_reason,
                response_language=response_language
            )
            raw_answer = generate(repair_prompt)
            citation_validation = validate_generated_answer(raw_answer)
            answer = citation_validation["answer"]
            language_validation = validate_answer_language(
                answer,
                response_language
            )
            answer_coverage_validation = validate_answer_coverage(
                answer,
                parsed_intents,
                response_language
            )
            emit(
                "answer_repair",
                "completed",
                "回答已重新生成并再次检查引用与语言",
                citation_valid=citation_validation["valid"],
                language_valid=language_validation["valid"]
            )

        if (
            validate_claims is None
            and not effective_validate_claims
            and _answer_contains_pharmacological_claim(answer)
        ):
            effective_validate_claims = True
            claim_validation_mode = "auto_output_content"
            parsed["claim_validation_mode"] = claim_validation_mode
            parsed["claim_validation_enabled"] = True

        if (
            citation_validation["valid"]
            and language_validation["valid"]
            and effective_validate_claims
        ):
            claim_validation = run_claim_validation(answer)

            # Rule-safe failures are handled by the supported-only builder:
            # uncited summaries are removed, unrelated extra citations are
            # dropped, and section-only errors are regrouped locally.  Spend
            # another generation only when a partly supported statement can
            # actually be narrowed using at least one supporting reference.
            semantically_repairable_claims = [
                item for item in claim_validation.get("claims", [])
                if item.get("verdict") == "partially_supported"
                and not item.get("citation_only_failure")
                and (
                    item.get("supporting_reference_ids")
                    or "supporting_reference_ids" not in item
                )
            ]
            should_repair_claims = (
                not claim_validation["valid"]
                and not claim_validation.get("judge_error")
                and generation_attempts < 2
                and bool(semantically_repairable_claims)
            )

            if should_repair_claims:
                repair_reason = (
                    "One or more scientific claims were not fully supported "
                    "by their cited study evidence."
                )
                emit(
                    "answer_repair",
                    "running",
                    "声明校验未通过，正在收窄或修复回答",
                    reason="claim_validation"
                )
                repair_prompt = build_answer_repair_prompt(
                    base_prompt=prompt,
                    previous_answer=answer,
                    reason=repair_reason,
                    claim_validation=claim_validation,
                    response_language=response_language
                )
                raw_answer = generate(repair_prompt)
                citation_validation = validate_generated_answer(raw_answer)
                answer = citation_validation["answer"]
                language_validation = validate_answer_language(
                    answer,
                    response_language
                )
                answer_coverage_validation = validate_answer_coverage(
                    answer,
                    parsed_intents,
                    response_language
                )
                previous_claim_validation = claim_validation
                claim_validation = (
                    run_claim_validation(
                        answer,
                        previous_validation=previous_claim_validation
                    )
                    if citation_validation["valid"]
                    and language_validation["valid"]
                    else None
                )
                emit(
                    "answer_repair",
                    "completed",
                    "修复回答已完成最终校验",
                    citation_valid=citation_validation["valid"],
                    language_valid=language_validation["valid"],
                    claim_valid=(
                        claim_validation.get("valid")
                        if claim_validation else None
                    )
                )

            if (
                claim_validation
                and not claim_validation["valid"]
                and not claim_validation.get("judge_error")
                and citation_validation["valid"]
                and language_validation["valid"]
            ):
                fallback = build_supported_claims_answer(
                    claim_validation,
                    references=references,
                    requested_intents=parsed_intents,
                    section_labels={
                        intent: intent_section_label(
                            intent,
                            response_language
                        )
                        for intent in parsed_intents
                    },
                    response_language=response_language
                )
                if fallback["answer"]:
                    original_validation = claim_validation
                    answer = fallback["answer"]
                    citation_validation = validate_generated_answer(answer)
                    answer = citation_validation["answer"]
                    language_validation = validate_answer_language(
                        answer,
                        response_language
                    )
                    answer_coverage_validation = validate_answer_coverage(
                        answer,
                        parsed_intents,
                        response_language
                    )
                    kept_ids = set(fallback["kept_claim_ids"])
                    kept_claims = []
                    for item in original_validation["claims"]:
                        if item.get("claim_id") not in kept_ids:
                            continue
                        kept_claims.append({
                            **item,
                            "verdict": "supported",
                            "citations": item.get(
                                "supporting_reference_ids", []
                            ),
                            "unsupported_reference_ids": [],
                            "citation_only_failure": False,
                            "section_intent": fallback.get("claim_sections", {}).get(item["claim_id"]),
                            "assigned_intent": fallback.get("claim_sections", {}).get(item["claim_id"]),
                            "section_valid": True,
                            "section_reason": None
                        })
                    claim_validation = {
                        **original_validation,
                        "valid": True,
                        "total_claims": len(kept_claims),
                        "supported_claims": len(kept_claims),
                        "partially_supported_claims": 0,
                        "unsupported_claims": 0,
                        "unverified_claims": 0,
                        "claims": kept_claims,
                        "safe_fallback_applied": True,
                        "removed_claim_ids": fallback[
                            "removed_claim_ids"
                        ],
                        "pre_fallback_counts": {
                            "total_claims": original_validation[
                                "total_claims"
                            ],
                            "supported_claims": original_validation[
                                "supported_claims"
                            ],
                            "partially_supported_claims": (
                                original_validation[
                                    "partially_supported_claims"
                                ]
                            ),
                            "unsupported_claims": original_validation[
                                "unsupported_claims"
                            ],
                            "unverified_claims": original_validation[
                                "unverified_claims"
                            ]
                        }
                    }
                    safe_fallback_applied = True
                    emit(
                        "answer_sanitization",
                        "completed",
                        "已只保留通过证据校验的声明",
                        kept_claim_count=len(kept_claims),
                        removed_claim_ids=fallback[
                            "removed_claim_ids"
                        ]
                    )

            if (
                claim_validation
                and claim_validation["valid"]
                and answer_coverage_validation
                and not answer_coverage_validation["valid"]
            ):
                fallback = build_supported_claims_answer(
                    claim_validation,
                    references=references,
                    requested_intents=parsed_intents,
                    section_labels={
                        intent: intent_section_label(
                            intent,
                            response_language
                        )
                        for intent in parsed_intents
                    },
                    response_language=response_language
                )
                if fallback["answer"]:
                    answer = fallback["answer"]
                    citation_validation = validate_generated_answer(answer)
                    answer = citation_validation["answer"]
                    language_validation = validate_answer_language(
                        answer,
                        response_language
                    )
                    answer_coverage_validation = validate_answer_coverage(
                        answer,
                        parsed_intents,
                        response_language
                    )
                    coverage_fallback_applied = True
                    kept_ids = set(fallback["kept_claim_ids"])
                    kept_claims = [{
                        **item,
                        "section_intent": fallback.get("claim_sections", {}).get(item["claim_id"]),
                        "assigned_intent": fallback.get("claim_sections", {}).get(item["claim_id"]),
                        "section_valid": True,
                        "section_reason": None,
                    } for item in claim_validation["claims"] if item.get("claim_id") in kept_ids]
                    claim_validation = {
                        **claim_validation,
                        "valid": bool(kept_claims),
                        "claims": kept_claims,
                        "total_claims": len(kept_claims),
                        "supported_claims": len(kept_claims),
                        "removed_claim_ids": list(dict.fromkeys([
                            *claim_validation.get("removed_claim_ids", []),
                            *fallback["removed_claim_ids"]])),
                    }
                    emit(
                        "answer_structuring",
                        "completed",
                        "已按问题的全部意图重新组织通过核验的声明",
                        required_intents=parsed_intents
                    )

        # Plant-source rows have already passed relation, botanical-name and
        # verbatim-text checks in build_source_answer. Preserve that stable
        # section in compound multi-intent answers even when the probabilistic
        # claim judge rejects or omits it during a repair pass.
        if verified_source_answer and len(parsed_intents) > 1:
            answer = _replace_markdown_section(
                answer,
                verified_source_answer,
                "plant_source"
            )
            citation_validation = validate_generated_answer(answer)
            answer = citation_validation["answer"]
            language_validation = validate_answer_language(
                answer,
                response_language
            )
            answer_coverage_validation = validate_answer_coverage(
                answer,
                parsed_intents,
                response_language
            )
            retrieval_stats["rule_grounded_sections"] = [
                "plant_source"
            ]

        emit(
            "answer_generation",
            "completed",
            "回答生成完成，引用和声明格式已检查",
            citation_valid=citation_validation["valid"],
            language_valid=language_validation["valid"],
            claim_validation_enabled=effective_validate_claims,
            claim_valid=(
                claim_validation.get("valid")
                if claim_validation else None
            ),
            coverage_valid=(
                answer_coverage_validation.get("valid")
                if answer_coverage_validation else None
            ),
            generation_attempts=generation_attempts,
            cited_reference_count=len(
                citation_validation["cited_reference_ids"]
            )
        )

        if not citation_validation["valid"]:
            status = "citation_warning"
            message = "模型回答在自动修复后仍存在引用问题，请检查。"
        elif not language_validation["valid"]:
            status = "language_warning"
            message = "模型回答在自动修复后仍未使用问题语言，请检查。"
        elif claim_validation and not claim_validation["valid"]:
            status = "claim_warning"
            message = (
                "模型回答在自动修复后仍存在未被证据完全支持"
                "或无法校验的结论，请检查。"
            )
        elif (
            answer_coverage_validation
            and not answer_coverage_validation["valid"]
        ):
            status = "coverage_warning"
            message = "回答未完整覆盖问题中的全部子问题，请检查。"
        else:
            status = "ok"
            message = None

    retrieval_stats["generation_attempts"] = generation_attempts
    retrieval_stats["answer_repaired"] = generation_attempts > 1
    retrieval_stats["repair_reason"] = repair_reason
    retrieval_stats["safe_fallback_applied"] = safe_fallback_applied
    retrieval_stats["coverage_fallback_applied"] = coverage_fallback_applied

    return {
        "status": status,
        "message": message,
        "question": question,
        "model": model,
        "compound": compound,
        "parsed": parsed,
        "candidate_edges": candidate_edges,
        "candidate_evidence": candidate_evidence,
        "relevant_edges": relevant_edges,
        "relevant_evidence": relevant_evidence,
        "references": references,
        "prompt": prompt,
        "answer": answer,
        "citation_validation": citation_validation,
        "language_validation": language_validation,
        "claim_validation": claim_validation,
        "answer_coverage_validation": answer_coverage_validation,
        "retrieval": retrieval_stats
    }


if __name__ == "__main__":

    question = "姜黄素有哪些抗肿瘤作用机制？"

    result = run_qa(
        question=question,
        call_model=False
    )

    print("Question:", result["question"])
    print("Compound:", result["compound"])

    print(
        "Candidate edges:",
        len(result["candidate_edges"])
    )

    print(
        "Candidate evidence:",
        len(result["candidate_evidence"])
    )

    print(
        "Relevant edges:",
        len(result["relevant_edges"])
    )

    print(
        "Relevant evidence:",
        len(result["relevant_evidence"])
    )

    print("\n" + "=" * 100)
    print("FINAL STRICT RAG PROMPT")
    print("=" * 100)

    print(result["prompt"])

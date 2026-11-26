import re


INTENT_SECTION_LABELS = {
    "compound_identity": {"zh": "是什么", "en": "What it is"},
    "plant_source": {"zh": "植物来源", "en": "Plant source"},
    "pharmacological_effect": {"zh": "功能与药理作用", "en": "Functions and pharmacological effects"},
    "anticancer_activity": {"zh": "抗肿瘤活性", "en": "Anticancer activity"},
    "anticancer_mechanism": {"zh": "抗肿瘤机制", "en": "Anticancer mechanisms"},
    "disease_association": {"zh": "疾病关联", "en": "Disease associations"},
    "target_mechanism": {"zh": "靶点与机制", "en": "Targets and mechanisms"},
    "toxicity": {"zh": "毒性与安全性", "en": "Toxicity and safety"},
    "pharmacokinetics": {"zh": "药代动力学", "en": "Pharmacokinetics"}
}


def intent_section_label(intent, response_language):
    language = "zh" if str(response_language or "").startswith("zh") else "en"
    labels = INTENT_SECTION_LABELS.get(intent)
    return labels[language] if labels else str(intent).replace("_", " ").title()


def validate_answer_coverage(answer, requested_intents, response_language):
    """Check mandatory section headings for a multi-intent answer."""

    intents = list(dict.fromkeys(requested_intents or []))
    if len(intents) <= 1:
        return {
            "valid": True,
            "required_intents": intents,
            "found_intents": intents,
            "missing_intents": []
        }

    cleaned_lines = []
    for line in str(answer or "").splitlines():
        cleaned = re.sub(r"^[\s#>*_`~\-\d.、)]+", "", line).strip()
        cleaned = re.sub(r"[*_`~]+", "", cleaned).strip().rstrip(":：")
        if cleaned:
            cleaned_lines.append(cleaned.casefold())

    found = []
    for intent in intents:
        label = intent_section_label(intent, response_language).casefold()
        if label in cleaned_lines:
            found.append(intent)

    missing = [intent for intent in intents if intent not in found]
    return {
        "valid": not missing,
        "required_intents": intents,
        "found_intents": found,
        "missing_intents": missing
    }


def build_qa_prompt(
    question,
    compound,
    edges,
    references,
    response_language=None,
    requested_intents=None,
    preferred_compound_name=None
):
    """
    把用户问题、最终结构化 references 和输出要求整理成生成提示词。

    支持材料只从 references 渲染；其中已包含关联关系元数据。
    edges 参数为旧调用兼容保留，不单独渲染原始图谱关系。
    不接受 context_builder 的候选调试文本。
    """

    lines = []

    lines.append(
        """You are an expert assistant for a natural-product knowledge database.

You are operating in STRICT RAG mode.

You must answer the user's question using ONLY the supporting literature study
units supplied below. Knowledge-graph relations are organization metadata, not
independent evidence.

Do NOT use your own pretrained knowledge to add factual information that is absent from the retrieved context.

Rules:
1. Every factual claim about the compound, target, pathway, biological effect, disease, toxicity, pharmacological activity, experimental result, or clinical finding must be supported by the supplied context.
2. Do not introduce mechanisms, targets, pathways, diseases, effects, or conclusions that are not present in the retrieved context.
3. If the retrieved evidence is insufficient to answer part or all of the question, explicitly state that the current retrieved evidence is insufficient.
4. Cite supporting literature references using [R1], [R2], [R3], etc.
5. Only use citation identifiers that actually appear in the supplied references.
6. Never invent or modify references, DOI, PMID, experimental results, numerical values, pathways, targets, or conclusions.
7. Do not attach a citation to a claim unless that reference's evidence specifically supports the claim.
8. Use the smallest sufficient set of references for each statement, normally one to three independently supporting studies. Do not attach every retrieved reference to a general claim.
9. Distinguish direct experimental findings from review-level, computational, docking, predictive, or other indirect evidence when this distinction is clear from the supplied context.
10. You may summarize, reorganize, compare, and explain the supplied evidence, but you may not add new factual knowledge from outside the retrieved context.
11. The retrieved knowledge graph may be used to organize the answer, but literature-supported factual claims should be grounded in the evidence supplied under [R] references.
12. Obey the Required Output Language section below. Do not switch languages merely because the retrieved evidence is written in another language.
13. Prefer a concise, structured, evidence-grounded answer over a broad but unsupported answer.
14. Citation format must be exactly [R1] or [R1][R2]. Do not write citation labels such as [文献支持：R1], grouped citations such as [R1, R2], bare numbers such as [1], or parenthesized forms such as (R1).
15. Place citations immediately after the claim they support. Example: Curcumin regulates the pathway [R1][R2].
16. Treat every [R] block as an independent study. Preserve its compound form or formulation, disease/model, cell line or organism, intervention, outcome, and direction of effect.
17. Never attribute a formulation result (for example, an encapsulated or nanoparticle preparation) to the free compound. Never turn a cell-line, animal, computational, or review finding into a clinical conclusion.
18. Do not fuse mechanisms from different papers into one causal statement. Report them study by study. You may synthesize studies only after describing compatible contexts and citing every supporting study.
19. If studies report different or opposite directions for the same target or pathway, state the findings separately and explicitly note the difference; do not average them into one mechanism.
20. A statement in a paper's discussion that describes an earlier study is secondary evidence. Label it as a prior report rather than an experiment performed by the current paper.
21. [K] identifiers are forbidden in the final answer. They are internal graph identifiers, not literature citations.
22. Answer only the requested intent. For an anticancer question, exclude mechanisms supported only by non-cancer studies, even when they involve a similarly named pathway.
23. Every factual sentence in an introduction, bullet, synthesis, or conclusion must carry its own supporting [R#] citation. Omit an uncited summary instead of repeating cited claims without citations.
24. Verify compound identity from each study's title and evidence text. A KG organizing relation does not prove that the study tested the main compound. Do not transfer findings from another compound, an unidentified numbered compound, a derivative, an analogue, a plant extract, or an analytical "compound equivalent" to the main compound.
25. Every citation attached to a claim must independently support that claim. Do not add an unrelated reference merely because another citation on the same sentence is valid. A compound detected only as an extract constituent or correlated chemical marker does not establish that the isolated compound caused the extract's activity.
26. The requested intents were retrieved independently. Organize the answer in one clearly separated section per requested intent, in the supplied order. Do not use a reference for an intent unless that reference is listed in that intent's evidence group.
27. Source evidence establishes botanical origin only. It does not establish a pharmacological effect. Pharmacological-effect evidence does not establish botanical origin unless the exact evidence item is explicitly assigned to both groups.
28. Do not add an uncited concluding sentence that repeats cited claims. Either cite the conclusion independently or omit it.
29. In the plant-source section, state each normalized botanical source concisely and use no more than three representative references for one source fact.
30. Preserve botanical names exactly as supplied, including author abbreviations and cultivar names. Do NOT translate a Latin botanical name into a Chinese plant name unless that Chinese name is explicitly provided in the evidence. Use the Latin name directly inside a Chinese sentence. This also applies to summaries: do not add inferred Chinese synonyms. Preserve uncertainty and do not add an unsupported "main source" ranking.
30. In the pharmacological-effect section, report at most six of the strongest distinct effects supported by the supplied evidence. Use one bullet for one effect; do not combine unrelated effects into a single claim.
31. Every pharmacological-effect bullet must carry its own one-to-three independently supporting references. Never place one shared citation block after a list of effects.
32. Preserve the evidence level in each effect claim: explicitly say review-level report, computational prediction, cell experiment, animal experiment, or clinical finding when the supplied context identifies it.
33. Prefer direct free-compound experiments over reviews, vague background statements, formulations, complexes, conjugates, and analogues. If only indirect evidence is available, label it instead of strengthening it.
34. The Main Compound value is authoritative entity-resolution metadata. You may state that canonical name without a citation, but any description of its class, chemistry, target, use, or properties still requires literature evidence.
35. Never omit a requested intent merely because its eligible-reference list is empty. Keep its required heading and explicitly state that the current ATNP retrieval did not find evidence for that part; do not fill the gap with pretrained knowledge.
36. Use the Preferred Display Name for the queried compound throughout the answer. Never repeat an unresolved or misspelled compound mention from the user question.
37. Put each claim in exactly one best-matching requested section. The compound-identity section is only for definition, class, form, precursor/prodrug, or identity distinctions; move effects, outcomes, and mechanisms to their corresponding sections. Do not repeat the same claim in multiple sections.
38. The plant-source section describes botanical origins and plant parts, not pharmacological properties. A paper or evidence sentence may describe both source and effects; put each separate claim in its matching section rather than assigning every claim to all of that paper's intent groups.
39. Write one independently verifiable factual assertion per bullet or sentence. Do not join a supported observation and a broader interpretation with commas, semicolons, "therefore", or similar wording. This rule also applies to introductions and summaries.
"""
    )

    lines.append("")
    lines.append("=== User Question ===")
    lines.append(question)

    lines.append("")
    lines.append("=== Main Compound ===")
    lines.append(compound)
    if preferred_compound_name:
        lines.append(f"Preferred Display Name: {preferred_compound_name}")

    requested_intents = list(dict.fromkeys(
        requested_intents or []
    ))
    if not requested_intents:
        requested_intents = list(dict.fromkeys(
            intent
            for reference in references
            for intent in reference.get("intent_groups", [])
        ))

    if requested_intents:
        lines.append("")
        lines.append("=== Independently Retrieved Intent Evidence Groups ===")
        for intent in requested_intents:
            eligible_ids = [
                reference["id"]
                for reference in references
                if intent in reference.get("intent_groups", [])
            ]
            lines.append("")
            lines.append(f"[Intent: {intent}]")
            lines.append(
                "Eligible references: "
                + (", ".join(eligible_ids) if eligible_ids else "none")
            )
            lines.append(
                "Use only evidence items carrying this intent label when "
                "answering this part of the question."
            )
            if intent == "plant_source":
                lines.append(
                    "Write a concise source section; use no more than three "
                    "representative references for the same botanical source."
                )
            elif intent == "pharmacological_effect":
                lines.append(
                    "Write at most six distinct effect bullets. Each bullet "
                    "must contain one effect, its evidence level/context, and "
                    "its own one-to-three supporting references. Do not use "
                    "this section merely to repeat the compound definition."
                )
            elif intent == "compound_identity":
                lines.append(
                    "Explain what the compound is using the strongest "
                    "classification or definition evidence. Distinguish the "
                    "compound from a salt, active metabolite, formulation, "
                    "or related drug when the evidence makes that distinction. "
                    "Do not list pharmacological effects in this section."
                )

    lines.append("")
    lines.append("=== Independent Literature Study Units ===")

    if references:
        for reference in references:

            lines.append("")
            lines.append(f"[Study {reference['id']}]")

            if reference.get("title"):
                lines.append(
                    f"Title: {reference['title']}"
                )

            if reference.get("doi"):
                lines.append(
                    f"DOI: {reference['doi']}"
                )

            if reference.get("pmid"):
                lines.append(
                    f"PMID: {reference['pmid']}"
                )

            hints = reference.get("evidence_type_hints") or []
            if hints:
                lines.append(
                    "Evidence type hints: " + ", ".join(hints)
                )

            lines.append(
                "Scope rule: Claims from this study must retain the exact "
                "experimental/formulation context stated in its evidence."
            )

            if reference.get("intent_groups"):
                lines.append(
                    "Eligible intent groups in this study: "
                    + ", ".join(reference["intent_groups"])
                )

            for index, item in enumerate(
                reference.get("evidence", []),
                1
            ):
                lines.append("")
                lines.append(f"Evidence {index}:")

                if item.get("section"):
                    lines.append(
                        f"Section: {item['section']}"
                    )

                if item.get("intent_groups"):
                    lines.append(
                        "Evidence intent groups: "
                        + ", ".join(item["intent_groups"])
                    )

                relations = item.get("relations") or [{
                    "relation": item.get("relation"),
                    "target": item.get("target"),
                    "target_type": item.get("target_type")
                }]
                for relation in relations:
                    relation_text = (
                        "Organizing relation: "
                        f"{compound} -- "
                        f"{relation.get('relation') or 'unknown'} --> "
                        f"{relation.get('target') or 'unknown'} "
                        f"[{relation.get('target_type') or 'unknown'}]"
                    )
                    lines.append(relation_text)

                    plant = relation.get("plant_entity")
                    if plant:
                        lines.append(
                            "Normalized botanical entity: "
                            f"scientific_name={plant.get('scientific_name')}; "
                            f"authority={plant.get('taxonomic_authority') or 'N/A'}; "
                            "plant_parts="
                            f"{', '.join(plant.get('plant_parts') or []) or 'N/A'}"
                        )

                if item.get("sentence"):
                    lines.append(
                        f"Evidence text: {item['sentence']}"
                    )
    else:
        lines.append("No relevant supporting literature references were retrieved.")

    if response_language is None:
        from app.language_validator import detect_question_language
        response_language = detect_question_language(question)

    from app.language_validator import language_instruction

    lines.append("")
    lines.append("=== Required Output Language ===")
    lines.append(language_instruction(response_language))
    lines.append(
        "This instruction is mandatory and takes priority over the language "
        "used by titles and evidence texts above."
    )

    if requested_intents:
        is_chinese = str(response_language or "").startswith("zh")
        lines.append("")
        lines.append("=== Mandatory Answer Coverage Plan ===")
        lines.append(
            "Include every section below in exactly this order. Use the exact "
            "Markdown heading shown. Do not merge, rename, or omit a section."
        )
        for intent in requested_intents:
            label = intent_section_label(intent, response_language)
            eligible_ids = [
                reference["id"]
                for reference in references
                if intent in reference.get("intent_groups", [])
            ]
            lines.append(f"### {label}")
            if intent == "plant_source":
                lines.append(
                    "植物名称必须直接使用证据中的原文（英文或拉丁文），不翻译植物名，"
                    "不添加中文别名或中文音译。每项格式：- 原文植物名：中文描述来源部位 [Rn]。"
                    "例如：- Citrus grandis (L.) Osbeck cv. Yongjiazaoxiangyou：果皮 [Rn]。"
                    "示例仅说明格式，实际名称和编号必须来自本次证据。"
                )
            if eligible_ids:
                lines.append(
                    "Answer this section using only: " + ", ".join(eligible_ids)
                )
            else:
                no_evidence_text = (
                    "当前 ATNP 检索证据不足以回答这一部分。"
                    if is_chinese else
                    "The current ATNP retrieval has insufficient evidence to answer this part."
                )
                lines.append(
                    "No eligible references were retrieved. State exactly: "
                    + no_evidence_text
                )

    lines.append("")
    lines.append("=== Answer ===")

    return "\n".join(lines)


def build_answer_repair_prompt(
    base_prompt,
    previous_answer,
    reason,
    claim_validation=None,
    response_language=None
):
    """Ask the same model for one constrained repair without new knowledge."""

    lines = [
        base_prompt,
        "",
        "=== Mandatory Answer Repair ===",
        "The previous answer failed strict validation. Repair it once using "
        "only the study units above.",
        f"Failure reason: {reason}",
        "Do not defend or discuss the previous answer. Return only the repaired "
        "answer, with exact [R#] citations. Remove every unrelated or uncited "
        "factual sentence. Use only supporting_reference_ids for each failed "
        "claim and remove every unsupported_reference_id. Do not add an "
        "uncited summary or conclusion. Copy every already-supported claim "
        "verbatim, including its citations and section; edit only the failed "
        "claims listed below. This allows unchanged claims to reuse their "
        "completed evidence validation."
    ]

    if response_language is not None:
        from app.language_validator import language_instruction
        lines.extend([
            "The repaired answer must use this language:",
            language_instruction(response_language)
        ])

    if claim_validation:
        failed_claims = [
            item
            for item in claim_validation.get("claims", [])
            if item.get("verdict") != "supported" or item.get("section_valid") is False
        ]
        if failed_claims:
            lines.append("Claims that must be removed, narrowed, or corrected:")
            for item in failed_claims:
                lines.append(
                    f"- {item.get('claim_id')}: {item.get('claim')} "
                    f"({item.get('verdict')}: {item.get('reason')}; "
                    f"section correction={item.get('section_reason') or 'none'}; "
                    f"suggested section={item.get('assigned_intent')}; "
                    "keep only supporting references="
                    f"{item.get('supporting_reference_ids') or []}; "
                    "remove references="
                    f"{item.get('unsupported_reference_ids') or []})"
                )

    lines.extend([
        "",
        "Previous invalid answer:",
        str(previous_answer or ""),
        "",
        "Repaired answer:"
    ])
    return "\n".join(lines)

import re
import unicodedata

from app.entity_resolver import (
    load_compound_name_index,
    normalize_entity_term,
    resolve_compound_in_text,
    suggest_compound_entities,
    suggest_similar_compound_entities
)


# ============================================================
# Intent 配置
# ============================================================

INTENT_CONFIG = {
    "compound_identity": {
        "relations": [
            "has_pharmacological_effect",
            "has_function",
            "inhibits",
            "targets",
            "binds_to"
        ],
        "keywords": [
            "identity",
            "compound class",
            "drug class",
            "prodrug",
            "defined as"
        ]
    },
    "anticancer_activity": {
        "relations": [
            "has_antitumor_activity"
        ],
        "keywords": [
            "anticancer",
            "antitumor",
            "cancer",
            "tumor",
            "tumour",
            "carcinoma"
        ]
    },
    "anticancer_mechanism": {
        "relations": [
            "inhibits",
            "activates",
            "regulates",
            "targets",
            "binds_to",
            "has_antitumor_activity"
        ],
        "keywords": [
            "cancer",
            "tumor",
            "tumour",
            "carcinoma",
            "glioblastoma",
            "apoptosis",
            "proliferation",
            "metastasis",
            "angiogenesis",
            "oncogenic"
        ]
    },

    "pharmacological_effect": {
        "relations": [
            "has_pharmacological_effect",
            "has_function",
            "inhibits",
            "activates",
            "regulates"
        ],
        "keywords": [
            "pharmacological",
            "effect",
            "activity",
            "therapeutic"
        ]
    },

    "disease_association": {
        "relations": [
            "associated_with",
            "has_pharmacological_effect",
            "has_antitumor_activity"
        ],
        "keywords": [
            "disease",
            "disorder",
            "condition",
            "cancer",
            "associated",
            "treatment",
            "therapeutic"
        ]
    },

    "target_mechanism": {
        "relations": [
            "targets",
            "binds_to",
            "regulates",
            "inhibits",
            "activates"
        ],
        "keywords": [
            "target",
            "protein",
            "gene",
            "pathway",
            "mechanism"
        ]
    },

    "plant_source": {
        "relations": [
            "isolated_from",
            "produced_by",
            "biosynthesized_in",
            "extracted_from",
            "derived_from",
            "contains"
        ],
        "keywords": [
            "plant",
            "source",
            "isolated",
            "extracted",
            "species"
        ]
    },

    "toxicity": {
        "relations": [
            "has_toxic_effect"
        ],
        "keywords": [
            "toxicity",
            "toxic",
            "adverse",
            "side effect"
        ]
    },

    "pharmacokinetics": {
        "relations": [
            "has_pharmacokinetic_property"
        ],
        "keywords": [
            "pharmacokinetic",
            "absorption",
            "distribution",
            "metabolism",
            "excretion",
            "bioavailability"
        ]
    }
}


REFERENTIAL_OR_GENERIC_MENTIONS = {
    "它",
    "其",
    "该化合物",
    "这个化合物",
    "这种化合物",
    "该天然产物",
    "这个天然产物",
    "这种天然产物",
    "该物质",
    "这个物质",
    "这种物质",
    "这个",
    "这种",
    "还",
    "另外",
    "此外",
    "除此之外",
    "作用",
    "功能",
    "功效",
    "毒性",
    "副作用",
    "药理作用",
    "靶点",
    "通路",
    "机制",
    "来源"
}
REFERENTIAL_ENTITY_PREFIX = re.compile(
    r"^(?:它|其|该化合物|这个化合物|这种化合物|该天然产物|"
    r"这个天然产物|这种天然产物|该物质|这个物质|这种物质|这个|这种)"
    r"(?:的|是|有|具有|来源|来自|提取|分离|作用|功能|靶点|通路|机制|毒性)",
    re.IGNORECASE
)


def normalize_text(text):
    """对问题和实体名称做统一的 Unicode 与大小写标准化。"""

    return normalize_entity_term(text)


def _contains_name(text, name):
    """
    判断标准化后的文本是否包含实体名称。

    ASCII 名称使用词边界，避免短名称命中普通英文单词的一部分；
    中文别名使用子串匹配。
    """

    if not name:
        return False

    if name.isascii():
        pattern = rf"(?<![0-9a-z]){re.escape(name)}(?![0-9a-z])"
        return re.search(pattern, text) is not None

    return name in text


def load_compound_names():
    """
    从 DuckDB nodes 表读取所有 Compound 节点名称。

    返回：
        [
            ("Curcumin", "curcumin"),
            ("Quercetin", "quercetin"),
            ...
        ]

    使用缓存，避免每次问题都重新读取数据库。
    """

    compounds = [
        (candidate["canonical_name"], canonical_key)
        for canonical_key, candidate in load_compound_name_index()[
            "compounds"
        ].items()
    ]

    # 长名称优先，避免短名称提前匹配
    compounds.sort(
        key=lambda x: len(x[0]),
        reverse=True
    )

    return compounds


def load_compound_aliases():
    """
    从 CSV 读取化合物别名到数据库标准名称的映射。

    较长别名优先匹配，避免较短别名抢先命中。
    """

    aliases = []
    index = load_compound_name_index()

    for alias, candidate_map in index["term_candidates"].items():
        for candidate in candidate_map.values():
            sources = candidate["terms"].get(alias, set())

            if sources == {"canonical"}:
                continue

            aliases.append((alias, candidate["canonical_name"]))

    aliases.sort(
        key=lambda item: len(item[0]),
        reverse=True
    )

    return aliases


def detect_compound(question):
    """
    从用户问题中识别数据库已有的 Compound。

    先匹配中英文别名，再匹配 nodes 表中的 Compound 标准名称。

    返回数据库中的标准名称。
    """

    resolution = resolve_compound_in_text(question)
    if resolution["status"] == "resolved":
        return resolution["canonical_name"]

    return None


def detect_ambiguous_compound(question):
    """Return all canonical candidates when an alias is one-to-many."""

    resolution = resolve_compound_in_text(question)
    if resolution["status"] != "ambiguous":
        return None

    return {
        "mention": resolution["mention"],
        "candidates": [
            item["canonical_name"]
            for item in resolution["candidates"]
        ]
    }


def extract_unresolved_compound_mention(question):
    """
    Detect an explicit-looking entity at the start of a question.

    This is only a safety guard. It prevents a missing alias from silently
    inheriting the previous conversation compound; it does not invent a
    canonical database entity.
    """

    text = unicodedata.normalize("NFKC", str(question or "")).strip()

    if not text:
        return None

    match = re.match(
        r"^\s*([0-9A-Za-z\u3400-\u9fffα-ωΑ-Ω·'′()\-/ ]{1,80}?)"
        r"(?=是什么|是啥|属于什么|指什么|指的是|有哪些|有什么|有何|"
        r"的|与|对|能否|是否|如何|会不会|可否|在)",
        text
    )

    if not match:
        return None

    mention = match.group(1).strip(" ,，。？?：:")
    normalized_mention = normalize_text(mention)

    if not mention or normalized_mention in {
        normalize_text(item)
        for item in REFERENTIAL_OR_GENERIC_MENTIONS
    }:
        return None

    # The task router may return the whole referential question as an entity,
    # then append "有什么作用" before parsing it again.  Never turn
    # "它来源于哪些植物" into a new compound merely because it now precedes
    # that appended intent phrase.
    if REFERENTIAL_ENTITY_PREFIX.search(normalized_mention):
        return None

    if len(mention) == 1 and not mention.isascii():
        return None

    return mention


def suggest_compounds(query, limit=10):
    """
    根据中英文别名和数据库标准名称提供化合物候选。

    排序优先级：完全匹配 > 前缀匹配 > 子串匹配。
    """

    return suggest_compound_entities(query=query, limit=limit)


def detect_intents(question):
    """返回问题中所有明确的意图，并保留主意图优先级。"""

    text = normalize_text(question)

    # --------------------------------------------------------
    # 1. 抗肿瘤机制
    # --------------------------------------------------------

    anticancer_words = [
        "抗肿瘤",
        "抗癌",
        "癌症",
        "肿瘤",
        "cancer",
        "tumor",
        "tumour",
        "anticancer",
        "antitumor"
    ]

    mechanism_words = [
        "机制",
        "作用机制",
        "通路",
        "靶点",
        "mechanism",
        "pathway",
        "target"
    ]

    has_cancer = any(
        word in text
        for word in anticancer_words
    )

    has_mechanism = any(
        word in text
        for word in mechanism_words
    )

    intents = []

    identity_words = [
        "是什么",
        "什么是",
        "属于什么",
        "是哪一类",
        "what is",
        "define",
        "definition",
        "which class"
    ]

    # “来源是什么” asks for an attribute, not the compound's identity.
    identity_text = re.sub(
        r"(?:植物来源|来源|作用机制|药理活性|药理作用|生物活性|功能|作用|靶点|通路|机制)"
        r"\s*(?:分别)?\s*(?:是|为)\s*什么",
        "", text
    )
    identity_text = re.sub(
        r"\bwhat\s+is\s+(?:the\s+)?(?:plant\s+source|source|"
        r"mechanism|function|activity|target|pathway)\b",
        "", identity_text
    )
    if any(word in identity_text for word in identity_words):
        intents.append("compound_identity")

    if has_cancer and has_mechanism:
        intents.append("anticancer_mechanism")
    elif has_cancer:
        intents.append("anticancer_activity")

    # --------------------------------------------------------
    # 2. 疾病关联
    # --------------------------------------------------------

    disease_words = [
        "疾病",
        "病症",
        "适应症",
        "治疗哪些病",
        "与哪些疾病相关",
        "disease",
        "disorder",
        "indication",
        "associated with"
    ]

    if any(word in text for word in disease_words):
        intents.append("disease_association")

    # --------------------------------------------------------
    # 3. 植物来源
    # --------------------------------------------------------

    plant_words = [
    "来源",
    "植物来源",
    "来自",
    "来自哪些植物",
    "哪些植物",
    "什么植物",
    "从哪里分离",
    "从什么植物",
    "分离自",
    "提取自",
    "植物",
    "source",
    "plant",
    "isolated from",
    "extracted from",
    "derived from"
    ]

    if any(word in text for word in plant_words):
        intents.append("plant_source")

    # --------------------------------------------------------
    # 4. 毒性
    # --------------------------------------------------------

    toxicity_words = [
        "毒性",
        "副作用",
        "不良反应",
        "toxic",
        "toxicity",
        "adverse effect"
    ]

    if any(word in text for word in toxicity_words):
        intents.append("toxicity")

    # --------------------------------------------------------
    # 5. 药代动力学
    # --------------------------------------------------------

    pk_words = [
        "药代",
        "药代动力学",
        "吸收",
        "代谢",
        "生物利用度",
        "pharmacokinetic",
        "bioavailability",
        "absorption",
        "metabolism"
    ]

    if any(word in text for word in pk_words):
        intents.append("pharmacokinetics")

    # --------------------------------------------------------
    # 6. 靶点 / 通路 / 机制
    # --------------------------------------------------------

    target_words = [
        "靶点",
        "通路",
        "作用机制",
        "分子机制",
        "调控",
        "抑制",
        "激活",
        "结合",
        "target",
        "pathway",
        "mechanism",
        "regulate",
        "inhibit",
        "activate",
        "bind"
    ]

    if (
        any(word in text for word in target_words)
        and not (has_cancer and has_mechanism)
    ):
        intents.append("target_mechanism")

    # --------------------------------------------------------
    # 7. 药理作用 / 功效
    # --------------------------------------------------------

    effect_words = [
        "功效",
        "药理作用",
        "活性",
        "有什么用",
        "有什么作用",
        "有哪些作用",
        "有何作用",
        "有什么功能",
        "有哪些功能",
        "有何功能",
        "effect",
        "activity",
        "function"
    ]

    text_without_mechanism = re.sub(
        r"作用机制|mechanism(?:s)?\s+of\s+action",
        "",
        text
    )
    has_general_effect = (
        any(word in text for word in effect_words)
        or "作用" in text_without_mechanism
    )
    specific_only_intents = {
        "anticancer_mechanism",
        "anticancer_activity"
    }
    if has_general_effect and not (
        set(intents)
        and set(intents).issubset(specific_only_intents)
    ):
        intents.append("pharmacological_effect")

    if not intents:
        intents.append("pharmacological_effect")

    return list(dict.fromkeys(intents))


def detect_intent(question):
    """返回主意图；保留该函数以兼容现有调用方。"""

    return detect_intents(question)[0]


def parse_question(question):
    """
    解析用户自然语言问题。

    Returns
    -------
    dict
        {
            "question": ...,
            "compound": ...,
            "intent": ...,
            "relations": [...],
            "keywords": [...]
        }
    """

    if not question or not question.strip():
        raise ValueError("question cannot be empty")

    compound_resolution = resolve_compound_in_text(question)
    ambiguous_compound = None
    compound = None
    compound_node_id = None
    compound_node_ids = []
    unresolved_compound_mention = None
    compound_suggestions = []

    if compound_resolution["status"] == "resolved":
        compound = compound_resolution["canonical_name"]
        compound_node_id = compound_resolution["compound_node_id"]
        compound_node_ids = compound_resolution["compound_node_ids"]
    elif compound_resolution["status"] == "ambiguous":
        ambiguous_compound = {
            "mention": compound_resolution["mention"],
            "candidates": [
                item["canonical_name"]
                for item in compound_resolution["candidates"]
            ]
        }
    else:
        unresolved_compound_mention = (
            extract_unresolved_compound_mention(question)
        )
        if unresolved_compound_mention:
            compound_suggestions = suggest_similar_compound_entities(
                unresolved_compound_mention
            )

    intents = detect_intents(question)
    intent = intents[0]
    intent_queries = []
    relations = []
    keywords = []

    for detected_intent in intents:
        config = INTENT_CONFIG.get(detected_intent, {})
        intent_relations = list(config.get("relations", []))
        intent_keywords = list(config.get("keywords", []))
        intent_queries.append({
            "intent": detected_intent,
            "relations": intent_relations,
            "keywords": intent_keywords
        })
        relations.extend(intent_relations)
        keywords.extend(intent_keywords)

    relations = list(dict.fromkeys(relations))
    keywords = list(dict.fromkeys(keywords))

    return {
        "question": question,
        "compound": compound,
        "compound_node_id": compound_node_id,
        "compound_node_ids": compound_node_ids,
        "compound_resolution": compound_resolution,
        "ambiguous_compound": ambiguous_compound,
        "unresolved_compound_mention": unresolved_compound_mention,
        "compound_suggestions": compound_suggestions,
        "intent": intent,
        "intents": intents,
        "intent_queries": intent_queries,
        "relations": relations,
        "keywords": keywords
    }


if __name__ == "__main__":

    compounds = load_compound_names()

    print(
        "Loaded compounds:",
        len(compounds)
    )

    print("\nFirst 20 compounds:")

    for name, _ in compounds[:20]:
        print(name)

    print("\n" + "=" * 80)

    test_questions = [
        "What are the pharmacological effects of Curcumin?",
        "What are the functions of Quercetin?",
        "What are the targets of Resveratrol?"
    ]

    for question in test_questions:

        result = parse_question(question)

        print("=" * 80)
        print("Question:", result["question"])
        print("Compound:", result["compound"])
        print("Intent:", result["intent"])
        print("Relations:", result["relations"])
        print("Keywords:", result["keywords"])

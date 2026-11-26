import re


CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


def detect_question_language(text):
    """Detect the supported answer language from the user's own wording."""

    value = str(text or "")
    return "zh" if len(CJK_PATTERN.findall(value)) >= 2 else "en"


def language_instruction(language):
    if language == "zh":
        return (
            "Simplified Chinese (简体中文). Write all explanatory prose in "
            "Chinese. Keep necessary Latin scientific names, gene symbols, "
            "pathway names, and exact [R#] citations unchanged."
        )

    return (
        "English. Write all explanatory prose in English. Keep necessary "
        "scientific names, symbols, and exact [R#] citations unchanged."
    )


def validate_answer_language(answer, expected_language):
    """Check that evidence language did not pull the answer off-language."""

    value = str(answer or "")
    cjk_count = len(CJK_PATTERN.findall(value))
    latin_count = len(LATIN_PATTERN.findall(value))
    letter_count = cjk_count + latin_count
    cjk_ratio = cjk_count / letter_count if letter_count else 0.0

    if expected_language == "zh":
        # Scientific Chinese answers can contain many Latin target and species
        # names, so a small but substantive Chinese proportion is sufficient.
        valid = cjk_count >= 4 and cjk_ratio >= 0.03
        detected_language = "zh" if valid else "en"
    else:
        valid = latin_count >= 4 and cjk_ratio < 0.25
        detected_language = "en" if valid else "zh"

    return {
        "valid": valid,
        "expected_language": expected_language,
        "detected_language": detected_language,
        "cjk_character_count": cjk_count,
        "latin_character_count": latin_count,
        "cjk_ratio": round(cjk_ratio, 4)
    }

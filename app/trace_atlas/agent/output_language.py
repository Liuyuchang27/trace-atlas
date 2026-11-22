"""Explicit workbench output languages, separate from legacy prompt modes."""

_LANGUAGES = {
    "zh-Hans": ("cn", "Simplified Chinese (简体中文)"),
    "zh-Hant": ("cn", "Traditional Chinese (繁體中文)"),
    "en-US": ("en", "English"),
}


def prompt_mode(language):
    return _LANGUAGES.get(language, (language, ""))[0]


def output_instruction(language):
    if language not in _LANGUAGES:
        return ""
    _, name = _LANGUAGES[language]
    return (
        f"\n\nOutput language: write all reader-facing text, headings and task descriptions in {name}, "
        "even if the question or evidence uses another language. Preserve original quotations, names, URLs, "
        "citations, JSON keys and enumerated values. Search in the languages best suited to the sources."
    )

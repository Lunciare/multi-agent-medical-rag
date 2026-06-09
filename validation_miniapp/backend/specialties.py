# The one canonical set used everywhere.
CANONICAL_SPECIALTIES = (
    "cardiology",
    "endocrinology",
    "gastroenterology",
    "infectious_diseases",
)

# Any known label (dataset OR orchestrator-emitted, any case) -> canonical.
_LABEL_TO_CANONICAL = {
    "cardiologist": "cardiology",
    "cardiology": "cardiology",
    "endocrinologist": "endocrinology",
    "endocrinology": "endocrinology",
    "gastroenterologist": "gastroenterology",
    "gastroenterology": "gastroenterology",
    "infectionist": "infectious_diseases",
    "infectious_diseases": "infectious_diseases",
    "infectious diseases": "infectious_diseases",
    "infectiologist": "infectious_diseases",
}

# Canonical -> Russian display label for the rater-facing routing screen (display only;
# stored data always keeps the canonical value).
RU_DISPLAY = {
    "cardiology": "Кардиолог",
    "endocrinology": "Эндокринолог",
    "gastroenterology": "Гастроэнтеролог",
    "infectious_diseases": "Инфекционист",
}


def to_canonical(label):
    """Map a dataset/emitted label to a canonical specialty, or None if unrecognised."""
    if label is None:
        return None
    return _LABEL_TO_CANONICAL.get(str(label).strip().lower())


def available_canonical(labels):
    """Map a list of labels (e.g. orchestrator.allowed_specialists) to canonical, in order,
    dropping any that don't map. Deduplicates while preserving order."""
    out = []
    for lbl in labels:
        c = to_canonical(lbl)
        if c is not None and c not in out:
            out.append(c)
    return out


def ru_display(canonical):
    """Canonical -> Russian display label (falls back to the input if unknown)."""
    return RU_DISPLAY.get(canonical, canonical)

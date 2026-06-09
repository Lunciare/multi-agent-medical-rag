import json
from pathlib import Path

from .specialties import CANONICAL_SPECIALTIES

GOLD_SPECIALTIES = CANONICAL_SPECIALTIES
_REQUIRED_STR_FIELDS = (
    "item_id", "case_ru", "answer_rag_ru", "answer_vanilla_ru", "routed_specialty",
)


class ItemsValidationError(Exception):
    """Raised when the items file is structurally invalid."""


def _non_empty_str(v):
    return isinstance(v, str) and v.strip() != ""


def validate_items(items):
    """Validate a parsed items list. Raise ItemsValidationError on first problem."""
    if not isinstance(items, list):
        raise ItemsValidationError("items file must be a JSON array")
    if not items:
        raise ItemsValidationError("items file is empty")

    seen_ids = set()
    for i, item in enumerate(items):
        where = f"item[{i}]"
        if not isinstance(item, dict):
            raise ItemsValidationError(f"{where}: must be a JSON object")
        # Prefer naming by item_id once we have it.
        iid = item.get("item_id")
        label = f"item_id={iid!r}" if _non_empty_str(iid) else where

        for field in _REQUIRED_STR_FIELDS:
            if not _non_empty_str(item.get(field)):
                raise ItemsValidationError(f"{label}: field '{field}' is missing or empty")

        if iid in seen_ids:
            raise ItemsValidationError(f"{label}: duplicate item_id")
        seen_ids.add(iid)

        gold = item.get("gold_specialty")
        if gold not in GOLD_SPECIALTIES:
            raise ItemsValidationError(
                f"{label}: gold_specialty must be one of {GOLD_SPECIALTIES}, got {gold!r}"
            )

        avail = item.get("available_specialties")
        if not isinstance(avail, list) or not avail or not all(_non_empty_str(s) for s in avail):
            raise ItemsValidationError(
                f"{label}: available_specialties must be a non-empty list of strings"
            )

    return items


def load_and_validate(items_path):
    """Read + parse + validate the items file. Returns (list, by_id_dict)."""
    path = Path(items_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ItemsValidationError(f"items file not found: {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ItemsValidationError(f"items file is not valid JSON: {e}")
    validate_items(data)
    return data, {it["item_id"]: it for it in data}

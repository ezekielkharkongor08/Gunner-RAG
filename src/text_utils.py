"""Small text helpers shared by ingestion, retrieval and the eval scripts.

Two jobs:

    * normalise season text on queries, e.g. ``2025-2026`` -> ``2025–26``
    * derive the extra section metadata (page type, season list)

Stored prose is never rewritten here. The normaliser is for queries only.
"""

from __future__ import annotations

import re

from src.config import settings

# Matches "2025-2026", "2025-26", "2025/26", "2025–2026", ...
_SEASON_RE = re.compile(settings.season_normalise_pattern)
_EN_DASH = "–"


def normalize_season_text(text: str) -> str:
    """Return ``text`` with season ranges in the Wikipedia form.

    ``2025-2026``, ``2025-26`` and ``2025/26`` all become ``2025–26``.
    Only the dash and the short year are changed; everything else is kept.
    """

    def _replace(match: re.Match) -> str:
        start, end = match.group(1), match.group(2)
        # A 4-digit end year in the same century is shortened to 2 digits.
        if len(end) == 4 and end[:2] == start[:2]:
            end = end[2:]
        return f"{start}{_EN_DASH}{end}"

    return _SEASON_RE.sub(_replace, text)


def find_seasons(text: str, *, limit: int | None = None) -> list[str]:
    """Return unique normalised seasons found in ``text``, in order.

    ``limit`` defaults to ``SEASONS_MAX`` from config.
    """
    cap = settings.seasons_max if limit is None else limit
    found: list[str] = []
    for match in _SEASON_RE.finditer(text):
        season = normalize_season_text(match.group(0))
        if season not in found:
            found.append(season)
        if len(found) >= cap:
            break
    return found


def page_type_for_title(title: str) -> str:
    """Classify a page title using ``page_type_rules`` from settings.yaml."""
    for rule in settings.page_type_rules:
        if re.search(str(rule["pattern"]), title):
            return str(rule["type"])
    return "other"

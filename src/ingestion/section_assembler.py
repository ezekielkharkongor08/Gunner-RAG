"""section_assembler.py -- clean up the section tree before splitting.

Runs between :mod:`document_cleaner` and :mod:`text_splitter`.

What it does, in this order:

    1. Drop boilerplate caption sections (notes, legends, "The following ...").
    2. Merge sections shorter than ``MIN_SECTION_CHARS``:
         * a parent's text goes to the start of its first child
         * otherwise it joins a sibling (previous first, then next)
         * a leftover under ``MIN_SECTION_KEEP`` is dropped
    3. Add the extra metadata used by retrieval and the eval:
       ``parent_section``, ``heading_level``, ``is_lead``, ``page_type``,
       ``seasons``, ``merged_from``, ``body_chars``, ``section_hash``.

Every drop and merge is logged, and a summary is returned so the caller can
print it. Text-only: no table handling here.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

from src.config import settings
from src.text_utils import find_seasons, page_type_for_title

logger = logging.getLogger(__name__)

INTRODUCTION = "Introduction"


# ---------------------------------------------------------------------------
# Small tree node
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    """One section document plus its place in the section tree."""

    heading: str
    path: tuple[str, ...]
    body: str
    doc: Document
    order: int = 0
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    dropped: bool = False
    drop_reason: str = ""

    @property
    def label(self) -> str:
        """Human-readable path used in logs and the header."""
        return " > ".join(self.path)

    @property
    def alive(self) -> bool:
        return not self.dropped and bool(self.body.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _path_of(doc: Document) -> tuple[str, ...]:
    """Return the section path as a tuple, falling back to the heading."""
    raw = str(doc.metadata.get("section_path", "")).strip()
    if raw:
        return tuple(part.strip() for part in raw.split(">") if part.strip())
    heading = str(doc.metadata.get("section", "")).strip()
    return (heading,) if heading else ("Untitled",)


def _is_boilerplate(body: str) -> tuple[bool, str]:
    """Return (drop?, reason) for a caption/legend section.

    A rule may carry ``keep_if_pattern``: when that regex is found anywhere in
    the body, the section is kept even though the drop rule matches. This is how
    the transfer spend/income totals survive ("The following ... Total
    expenditure: £248.2 million ...").
    """
    text = body.strip()
    for rule in settings.boilerplate_patterns:
        # Per-rule max_chars wins; BOILERPLATE_MAX_CHARS is the fallback.
        max_chars = int(rule.get("max_chars", settings.chunk.boilerplate_max_chars))
        if len(text) < max_chars and re.match(str(rule["pattern"]), text):
            keep_if = rule.get("keep_if_pattern")
            if keep_if and re.search(str(keep_if), text):
                logger.info(
                    "Kept section that matches %r because it contains %r.",
                    rule["pattern"], keep_if,
                )
                continue
            return True, f"boilerplate: {rule['pattern']}"
    return False, ""


def _merge_label(node: _Node) -> str:
    """Label kept in front of merged text, e.g. "Kent County FA: ..."."""
    return f"{node.heading}: {node.body.strip()}"


def _group_by_page(documents: list[Document]) -> list[tuple[str, str, list[Document]]]:
    """Group documents per page, keeping the original section order."""
    groups: dict[tuple[str, str], list[Document]] = {}
    for doc in documents:
        key = (
            str(doc.metadata.get("title", "")),
            str(doc.metadata.get("url", "")),
        )
        groups.setdefault(key, []).append(doc)
    return [(title, url, docs) for (title, url), docs in groups.items()]


def _build_tree(docs: list[Document]) -> list[_Node]:
    """Create nodes and link each one to the longest existing parent path."""
    nodes: list[_Node] = []
    by_path: dict[tuple[str, ...], _Node] = {}

    for index, doc in enumerate(docs):
        path = _path_of(doc)
        node = _Node(
            heading=path[-1],
            path=path,
            body=doc.page_content.strip(),
            doc=doc,
            order=index,
        )
        nodes.append(node)
        # Keep the first node for parent lookups if a path repeats.
        by_path.setdefault(path, node)

    for node in nodes:
        parent = None
        for length in range(len(node.path) - 1, 0, -1):
            candidate = by_path.get(node.path[:length])
            if candidate is not None:
                parent = candidate
                break
        node.parent = parent
        if parent is not None:
            parent.children.append(node)

    return nodes


def _siblings(node: _Node, groups: dict[tuple[str, ...], list[_Node]]) -> list[_Node]:
    """Return nodes that share ``node``'s logical parent path.

    The parent document may not exist (e.g. ``Competitions > Premier League``
    has no prose of its own), so we group by ``path[:-1]`` instead of relying
    on the parent node.
    """
    return groups[node.path[:-1]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def assemble_sections(documents: list[Document]) -> tuple[list[Document], dict]:
    """Assemble cleaned section documents. Returns (documents, report)."""
    report: dict = {
        "sections_in": len(documents),
        "sections_out": 0,
        "dropped": [],
        "merged": [],
    }

    if not documents:
        return [], report

    assembled: list[Document] = []
    min_chars = settings.chunk.min_section_chars
    keep_chars = settings.chunk.min_section_keep
    max_passes = settings.chunk.min_section_max_passes

    for title, _url, docs in _group_by_page(documents):
        nodes = _build_tree(docs)
        # Logical sibling groups: same parent path, even when the parent has
        # no text of its own.
        sibling_groups: dict[tuple[str, ...], list[_Node]] = {}
        for node in nodes:
            sibling_groups.setdefault(node.path[:-1], []).append(node)
        page_type = page_type_for_title(title)

        # --- 1. Boilerplate caption sections -----------------------------
        for node in nodes:
            if not node.body.strip():
                node.dropped, node.drop_reason = True, "empty"
                continue
            drop, reason = _is_boilerplate(node.body)
            if drop:
                node.dropped, node.drop_reason = True, reason
                logger.info(
                    "Dropped section %s [%s] (%d chars): %s",
                    node.label, title, len(node.body), node.body,
                )
                report["dropped"].append({
                    "page": title,
                    "path": node.label,
                    "chars": len(node.body),
                    "reason": reason,
                    "text": node.body,
                })

        # --- 2. Merge short sections -------------------------------------
        for _pass in range(max_passes):
            changed = False
            for node in nodes:
                if not node.alive:
                    continue
                # Never merge the lead: it is the page summary.
                if node.heading == INTRODUCTION or node.path == (INTRODUCTION,):
                    continue
                if len(node.body) >= min_chars:
                    continue

                # 2a. Parent text -> start of its first live child.
                first_child = next(
                    (child for child in node.children if child.alive), None
                )
                if first_child is not None:
                    first_child.body = (
                        f"{_merge_label(node)}\n\n{first_child.body.strip()}"
                    )
                    first_child.merged_from.insert(0, node.heading)
                    node.dropped, node.drop_reason = True, "merged into child"
                    changed = True
                    logger.info(
                        "Merged parent %s (%d chars) into child %s [%s]",
                        node.label, len(node.body), first_child.label, title,
                    )
                    report["merged"].append({
                        "page": title,
                        "from": node.label,
                        "into": first_child.label,
                        "chars": len(node.body),
                        "parts": [node.heading],
                        "how": "parent -> first child",
                    })
                    continue

                # 2b. Sibling merge: previous first, then next.
                siblings = _siblings(node, sibling_groups)
                position = siblings.index(node)
                previous = next(
                    (item for item in reversed(siblings[:position]) if item.alive),
                    None,
                )
                following = next(
                    (item for item in siblings[position + 1:] if item.alive),
                    None,
                )
                target = previous or following
                if target is not None:
                    text = _merge_label(node)
                    if target is previous:
                        target.body = f"{target.body.strip()}\n\n{text}"
                    else:
                        target.body = f"{text}\n\n{target.body.strip()}"
                    target.merged_from.append(node.heading)
                    node.dropped, node.drop_reason = True, "merged into sibling"
                    changed = True
                    logger.info(
                        "Merged sibling %s (%d chars) into %s [%s]",
                        node.label, len(node.body), target.label, title,
                    )
                    report["merged"].append({
                        "page": title,
                        "from": node.label,
                        "into": target.label,
                        "chars": len(node.body),
                        "parts": [node.heading],
                        "how": "sibling",
                    })
                    continue

                # 2c. A child with no sibling can still join its parent.
                if node.parent is not None and node.parent.alive:
                    target = node.parent
                    target.body = f"{target.body.strip()}\n\n{_merge_label(node)}"
                    target.merged_from.append(node.heading)
                    node.dropped, node.drop_reason = True, "merged into parent"
                    changed = True
                    logger.info(
                        "Merged child %s (%d chars) into parent %s [%s]",
                        node.label, len(node.body), target.label, title,
                    )
                    report["merged"].append({
                        "page": title,
                        "from": node.label,
                        "into": target.label,
                        "chars": len(node.body),
                        "parts": [node.heading],
                        "how": "child -> parent",
                    })
                    continue

                # 2d. No parent and no sibling: keep only if long enough.
                if len(node.body) < keep_chars:
                    node.dropped, node.drop_reason = True, "too short"
                    changed = True
                    logger.info(
                        "Dropped short section %s (%d chars) [%s]",
                        node.label, len(node.body), title,
                    )
                    report["dropped"].append({
                        "page": title,
                        "path": node.label,
                        "chars": len(node.body),
                        "reason": "too short (no sibling to merge)",
                        "text": node.body,
                    })
            if not changed:
                break

        # --- 3. Build output documents -----------------------------------
        for node in nodes:
            if not node.alive:
                continue
            body = node.body.strip()
            is_lead = node.heading == INTRODUCTION or node.path == (INTRODUCTION,)
            if settings.seasons_from_text:
                seasons_text = f"{title}\n{body}"
            metadata = dict(node.doc.metadata)
            seasons_source = (
                f"{title}\n{body}" if settings.seasons_from_text else title
            )
            metadata.update(
                section=node.heading,
                section_path=node.label,
                parent_section=node.path[-2] if len(node.path) > 1 else "",
                heading_level=0 if is_lead else len(node.path) + 1,
                is_lead=is_lead,
                page_type=page_type,
                seasons=find_seasons(seasons_source),
                merged_from=list(node.merged_from),
                body_chars=len(body),
                section_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            )
            assembled.append(Document(page_content=body, metadata=metadata))

    report["sections_out"] = len(assembled)
    logger.info(
        "Section assembly: %d in, %d out, %d dropped, %d merged.",
        report["sections_in"], report["sections_out"],
        len(report["dropped"]), len(report["merged"]),
    )
    return assembled, report


def summarize_report(report: dict) -> str:
    """One-line summary for run_ingestion's stdout."""
    return (
        f"sections in={report.get('sections_in', 0)} "
        f"out={report.get('sections_out', 0)} "
        f"dropped={len(report.get('dropped', []))} "
        f"merged={len(report.get('merged', []))}"
    )

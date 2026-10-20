"""End-to-end quality checks on a small offline fixture.

These are the rules we do not want to regress:

    * every chunk has the title + section header
    * no chunk starts with a bare ". "
    * no chunk body is shorter than CHUNK_MIN_SIZE
    * chunk ids are unique
    * short sections are merged, not dropped (Kits, Kent County FA)
    * boilerplate captions are gone
    * paragraph breaks survive
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.section_assembler import assemble_sections
from src.ingestion.text_splitter import split_documents

TITLE = "2025–26 Arsenal F.C. season"
URL = "https://en.wikipedia.org/wiki/2025%E2%80%9326_Arsenal_F.C._season"


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "wikipedia",
            "title": TITLE,
            "url": URL,
            "section": path.split(" > ")[-1],
            "section_path": path,
            "language": "en",
            "content_type": "prose",
        },
    )


def _fixture() -> list[Document]:
    long_lead = "\n\n".join(
        f"Paragraph {index} about the season. " + ("x" * 180)
        for index in range(1, 6)
    )
    may = (
        "Arsenal opened May with a 3–0 win over Fulham.\n\n"
        "On 19 May, Arsenal were crowned Premier League champions."
    )
    return [
        _doc("Introduction", long_lead),
        _doc("Kits", "Supplier: Adidas. Manufacturer: Adidas."),
        _doc(
            "Kits > Kit information",
            "Home kit: red body with white sleeves.\n\n"
            "Away kit: yellow body with blue sleeves.\n\n"
            "Third kit: navy body with gold trim, worn in European away "
            "matches during the league phase and the knockout rounds.",
        ),
        _doc("Review", "Review overview of the season. " * 6),
        _doc("Review > May", may),
        _doc(
            "Honours and achievements > Regional honours > London FA",
            "London Senior Cup winners. " * 12,
        ),
        _doc(
            "Honours and achievements > Regional honours > Kent County FA",
            "Kent Senior Cup winners.",
        ),
        _doc(
            "Statistics > Appearances",
            "The following 31 players made appearances for the club.",
        ),
        _doc(
            "Statistics > Goals",
            "Arsenal scored 121 goals in all competitions. " * 8,
        ),
        _doc(
            "First team > First team coaching staff",
            "Note: Age as of 30 June 2026.",
        ),
    ]


def _chunks() -> list[Document]:
    assembled, _report = assemble_sections(_fixture())
    return split_documents(assembled)


def test_every_chunk_has_a_header():
    for chunk in _chunks():
        section_path = chunk.metadata["section_path"]
        assert chunk.page_content.startswith(
            f"{TITLE}\nSection: {section_path}\n\n"
        )


def test_no_chunk_starts_with_a_dot():
    for chunk in _chunks():
        body = chunk.page_content.split("\n\n", 1)[1]
        assert not body.startswith(".")


def test_no_chunk_body_is_shorter_than_min_size():
    for chunk in _chunks():
        assert chunk.metadata["body_chars"] >= settings.chunk.min_size


def test_chunk_ids_are_unique():
    ids = [chunk.metadata["chunk_id"] for chunk in _chunks()]
    assert len(ids) == len(set(ids))


def test_kits_text_is_inside_kit_information():
    chunks = _chunks()
    kit_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["section_path"] == "Kits > Kit information"
    ]
    assert kit_chunks
    assert "Supplier: Adidas" in kit_chunks[0].page_content
    assert not any(
        chunk.metadata["section_path"] == "Kits" for chunk in chunks
    )


def test_kent_county_fa_is_not_standalone():
    paths = [chunk.metadata["section_path"] for chunk in _chunks()]
    assert not any(path.endswith("> Kent County FA") for path in paths)


def test_boilerplate_is_gone():
    text = "\n".join(chunk.page_content for chunk in _chunks())
    assert "The following 31 players" not in text
    assert "Note: Age as of" not in text


def test_paragraph_breaks_survive():
    may = [
        chunk for chunk in _chunks()
        if chunk.metadata["section_path"] == "Review > May"
    ][0]
    assert "\n\n" in may.page_content

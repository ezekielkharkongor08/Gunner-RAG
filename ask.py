"""Ask one question from the terminal.

    python ask.py "Who did Arsenal lose to in the 2025-26 EFL Cup final?"
    python ask.py --check "Who did Arsenal lose to in the 2025-26 EFL Cup final?"

Prints the answer, then the numbered sources (title, section, URL, fetched_at).
Sources are built in code, so the numbers match the ``[n]`` markers in the
answer. Chunks from the same section share one line.

With ``--check`` it also prints which numbers and names in the answer do not
appear in the sources (see src/generation/grounding.py).

Exit codes:
    0  an answer with at least one source
    1  no question given
    2  no indexed passage matched the question (nothing to cite)
    3  --check found a number in the answer that is not in the sources
"""

from __future__ import annotations

import re
import sys

from src.generation.answer_chain import answer_question, format_sources
from src.generation.grounding import check_answer

# Wording the model uses when the corpus does not answer the question.
REFUSAL = re.compile(
    r"(do(es)? not contain|not mentioned|no information|not (?:found|available|present)"
    r"|cannot (?:be )?answer|insufficient)",
    re.IGNORECASE,
)


def main(argv: list[str]) -> int:
    # Windows pipes default to a legacy code page; force UTF-8 so en dashes and
    # accents in titles survive `python ask.py ... | more`.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    question = " ".join(
        part for part in argv[1:] if part != "--check"
    ).strip()
    check = "--check" in argv[1:]
    if not question:
        print('Usage: python ask.py [--check] "your question"')
        return 1

    result = answer_question(question)

    print(f"\nQ: {question}\n")
    print((result.answer or "").strip() or "(empty answer)")

    if result.sources:
        label = (
            "Retrieved chunks (not used for the answer):"
            if REFUSAL.search(result.answer or "")
            else "Sources:"
        )
        print(f"\n{label}")
        print(format_sources(result.sources))

        if check:
            report = check_answer(
                result.answer or "",
                [chunk.body_text() for chunk in result.sources],
            )
            print("\nGrounding check (tokens below do not appear in the sources):")
            print("  unsupported numbers:", ", ".join(report["numbers"]) or "none")
            print("  unsupported names:  ", ", ".join(report["names"][:6]) or "none")
            if report["numbers"]:
                return 3
        return 0

    print("\nNo indexed passage matched this question.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

import json
import re

import fitz  # PyMuPDF
import ollama
from pydantic import BaseModel, Field

PDF_PATH = "./gerbaudo2026.pdf"
MODEL = "qwen3.8:27b-mlx"  # check exact tag with `ollama list`; try "qwen3.6:27b" for more accuracy
DPI = 200
USE_TEXT_LAYER = (
    True  # use embedded text when the PDF has it (faster, more accurate than OCR)
)
MIN_TEXT_CHARS = 200  # below this, a page is treated as scanned and sent to the VLM
CHUNK_CHARS = 5000  # size of each extraction chunk
OVERLAP_CHARS = (
    800  # overlap so entries cut at a chunk edge appear whole in the next chunk
)

CHAT_OPTIONS = {"temperature": 0, "num_ctx": 16384}


# ---------- Data model for structured output ----------


class BibEntry(BaseModel):
    authors: list[str] = Field(..., description="All authors, full names in order")
    year: int | None = None
    title: str
    source: str = Field(..., description="Journal, book title or publisher")
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    raw_entry: str = Field(..., description="The original entry as printed")


class Bibliography(BaseModel):
    entries: list[BibEntry]


# ---------- Step 1: PDF -> text/markdown ----------


def page_to_markdown(page: "fitz.Page") -> str:
    if USE_TEXT_LAYER:
        text = page.get_text().strip()
        if len(text) >= MIN_TEXT_CHARS:
            return text

    png = page.get_pixmap(dpi=DPI).tobytes("png")
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Transcribe this page to markdown exactly as printed. "
                    "Preserve every character, including accents, DOIs and page ranges. "
                    "Do not summarize, translate or add commentary."
                ),
                "images": [png],
            }
        ],
        think=False,
        options=CHAT_OPTIONS,
    )
    return resp["message"]["content"].strip()


# ---------- Step 2: text -> structured entries ----------


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on line boundaries into overlapping chunks."""
    lines = text.splitlines(keepends=True)
    chunks, start = [], 0
    while start < len(lines):
        end, total = start, 0
        while end < len(lines) and total < size:
            total += len(lines[end])
            end += 1
        chunks.append("".join(lines[start:end]))
        if end >= len(lines):
            break
        # step back far enough to cover the overlap
        back, carried = end, 0
        while back > start + 1 and carried < overlap:
            back -= 1
            carried += len(lines[back])
        start = back
    return chunks


def extract_entries(chunk: str, is_first: bool, is_last: bool) -> list[BibEntry]:
    edge_rules = []
    if not is_first:
        edge_rules.append(
            "ignore a reference that is cut off at the very start of the text"
        )
    if not is_last:
        edge_rules.append(
            "ignore a reference that is cut off at the very end of the text"
        )
    edge = ("Additionally, " + " and ".join(edge_rules) + ". ") if edge_rules else ""

    resp = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "The following text is part of an academic reference list. "
                    "Return one entry object per bibliographic reference, in order. "
                    "Copy raw_entry exactly as printed. Use null for fields that are "
                    "not present; never invent values. Skip headings, page numbers "
                    "and running headers. " + edge + "\n\n" + chunk
                ),
            }
        ],
        format=Bibliography.model_json_schema(),  # constrained decoding
        think=False,
        options=CHAT_OPTIONS,
    )
    return Bibliography.model_validate_json(resp["message"]["content"]).entries


def dedupe(entries: list[BibEntry]) -> list[BibEntry]:
    """Overlapping chunks can yield the same reference twice; keep the longer version."""
    seen: dict[str, int] = {}
    out: list[BibEntry] = []
    for e in entries:
        key = re.sub(r"\W+", "", e.raw_entry.lower())[:80]
        if key in seen:
            i = seen[key]
            if len(e.raw_entry) > len(out[i].raw_entry):
                out[i] = e
        else:
            seen[key] = len(out)
            out.append(e)
    return out


# ---------- Main ----------


def main():
    doc = fitz.open(PDF_PATH)

    # Step 1: raw markdown/text for the whole document
    page_texts = []
    for i, page in enumerate(doc, 1):
        print(f"Reading page {i}/{len(doc)}")
        page_texts.append(page_to_markdown(page))
    # Join with a single newline so an entry split across pages stays in one piece
    full_text = "\n".join(page_texts)
    with open("bibliography_raw.md", "w", encoding="utf-8") as f:
        f.write(full_text)
    print("Step 1 done: saved to bibliography_raw.md")

    # Step 2: structured entries from overlapping chunks
    chunks = chunk_text(full_text, CHUNK_CHARS, OVERLAP_CHARS)
    entries: list[BibEntry] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Extracting chunk {i}/{len(chunks)}")
        entries.extend(
            extract_entries(chunk, is_first=(i == 1), is_last=(i == len(chunks)))
        )

    bib = Bibliography(entries=dedupe(entries))
    with open("bibliography.json", "w", encoding="utf-8") as f:
        f.write(bib.model_dump_json(indent=2))

    print(f"Step 2 done: {len(bib.entries)} entries saved to bibliography.json")
    for e in bib.entries:
        first_author = e.authors[0] if e.authors else "?"
        print(f"  - {first_author} ({e.year}): {e.title}")


if __name__ == "__main__":
    main()

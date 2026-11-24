import base64
import json
import os
import time

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from pydantic import BaseModel, Field
from mistralai.extra import response_format_from_pydantic_model

load_dotenv()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

PDF_PATH = "./gerbaudo2026.pdf"
MODEL = "mistral-ocr-4-1"


# ---------- Data model for structured output ----------


class BibEntry(BaseModel):
    authors: list[str] = Field(..., description="All authors, full names in order")
    year: int
    title: str
    source: str = Field(..., description="Journal, book title or publisher")
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    raw_entry: str = Field(..., description="The original entry as printed")


class Bibliography(BaseModel):
    entries: list[BibEntry]


# ---------- Helpers ----------


def encode_pdf(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def document_chunk(pdf_b64: str) -> dict:
    return {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{pdf_b64}",
    }


def ocr_process_with_retry(client, max_retries: int = 5, **kwargs):
    """Call client.ocr.process with exponential backoff on 429 errors."""
    for attempt in range(max_retries):
        try:
            return client.ocr.process(**kwargs)
        except SDKError as e:
            if e.status_code != 429 or attempt == max_retries - 1:
                raise
            wait = (2**attempt) * 2  # 2, 4, 8, 16, 32 seconds
            print(
                f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait)


# ---------- Main ----------


def main():
    pdf_b64 = encode_pdf(PDF_PATH)

    # Step 1: basic OCR - raw markdown per page
    print("Step 1: OCR to markdown...")
    ocr_response = ocr_process_with_retry(
        client,
        model=MODEL,
        document=document_chunk(pdf_b64),
        include_image_base64=False,
    )
    full_markdown = "\n\n".join(page.markdown for page in ocr_response.pages)
    with open("bibliography_raw.md", "w", encoding="utf-8") as f:
        f.write(full_markdown)
    print(
        f"Step 1 done: {len(ocr_response.pages)} page(s), saved to bibliography_raw.md"
    )

    time.sleep(2)  # small pause to be gentle on rate limits

    # Step 2: structured annotations - parsed bibliography entries
    print("Step 2: structured extraction...")
    ocr_structured = ocr_process_with_retry(
        client,
        model=MODEL,
        document=document_chunk(pdf_b64),
        document_annotation_format=response_format_from_pydantic_model(Bibliography),
        document_annotation_prompt=(
            "This document contains an academic reference list. "
            "Return one entry object per bibliographic reference, in order."
        ),
        include_image_base64=False,
    )
    bib = Bibliography.model_validate_json(
        json.dumps(ocr_structured.document_annotation)
    )

    with open("bibliography.json", "w", encoding="utf-8") as f:
        f.write(bib.model_dump_json(indent=2))

    print(f"Step 2 done: {len(bib.entries)} entries, saved to bibliography.json")
    for e in bib.entries:
        first_author = e.authors[0] if e.authors else "?"
        print(f"  - {first_author} ({e.year}): {e.title}")


if __name__ == "__main__":
    main()

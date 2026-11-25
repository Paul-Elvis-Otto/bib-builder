> **Disclaimer:** This project was built with the help of GPT 6 Astra. It is experimental; review generated citations and identifier suggestions before using them.

# Bibliography Studio

A local, single-file Python web app for extracting bibliography entries from PDFs with an Ollama model.

- Upload a PDF and select bibliography pages using the numbered preview.
- Draw exclusion boxes over headers, footers, or other unwanted content.
- Review extracted citations, copy individual or combined BibTeX, and download BibTeX or RIS files.
- Optionally ask Gemini to assess citation metadata and suggest/check DOI or ISBN identifiers. Reviews show agreement, disagreement, or uncertainty without modifying citations or exports.

The app lives in `server.py`. `main.py` is the original minimal extraction example.

## Run locally

You need Python 3.12+, `uv`, and a running Ollama-compatible server with your chosen model available.

Copy the provided [`.env.example`](.env.example) to `.env` if you do not already have one, then adjust the model name and server settings:

```bash
cp -n .env.example .env
```

Leave `GEMINI_API_KEY` empty to keep Gemini review disabled. Your local `.env` is ignored by Git; never commit API keys.

Install dependencies and start the app:

```bash
uv sync
uv run python server.py
```

Open **http://127.0.0.1:8000**. Stop the server with `Ctrl+C`.

The default extraction model is `qwen3.8:27b-mlx`; change it to the exact model name available on your device. The model must support structured JSON output and, for scanned or low-text pages, image input. Set `OLLAMA_HOST` in `.env` if your model server uses a different address.

## Switch models

Near the top of `server.py`, replace the default model names in these lines:

```python
MODEL = os.getenv("BIB_MODEL", "qwen3.8:27b-mlx")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-4-hydrogen")
```

Alternatively, set `BIB_MODEL` or `GEMINI_MODEL` in a `.env` file beside `server.py`. Environment settings override the defaults in the script. Restart the server after changing them.

## Optional Gemini review

Add your key to `.env`:

```dotenv
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-4-hydrogen
```

Restart the app, extract citations, then click **Review with Gemini** beside an entry. Each review sends citation metadata and the original reference text—not the PDF—to Google, using search-grounded research followed by a structured assessment.

An API key is required even for free-tier usage. The current review uses Google Search grounding, which requires the paid tier for the default model; its included search allowance does not make model tokens free. Check [Google's current pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-4-hydrogen) before enabling it. Keep your API key private.

Uploads and results are temporary, held in server memory, and lost on restart. Separate database-verification functions are placeholders; Gemini reviews are model assessments, not guaranteed verification.

## Mistral + OpenAlex citation graph

Run the alternative browser app with:

```bash
uv run python mistral.py
```

Set `MISTRAL_API_KEY` and `OPENALEX_API_KEY` in `.env` first. OpenAlex requests
use the key in the Authorization Bearer header; missing keys produce an error
instead of anonymous requests. Restart the app after changing `.env`.
Optional settings are `MISTRAL_OCR_MODEL`, `MISTRAL_CHAT_MODEL`, and `BIB_DATABASE` (default
`papers.sqlite3`). It uses the same host/port and browser controls as `server.py`;
run one app at a time or set a different `BIB_PORT`.

Paper identification sends only physical page one initially, expanding to the
first 2, 4, and further pages only when Mistral returns no identifiable title.
Leave bibliography pages blank (or enter `auto`) to search the PDF's local text
for References, Bibliography, Literaturverzeichnis, or equivalent heading lines.
Only pages from the earliest plausible heading onward are initially sent for
reference detection; keeping the remaining pages avoids cutting off continuations.
If no heading is found (including scanned PDFs), no section is confirmed, or no
entries are extracted, the pipeline falls back to scanning all pages as before.
OCR text already obtained is reused within the job. A nonempty extraction does
not prove completeness; review results and use manual pages for unusual documents.

Manual page ranges bypass automatic reference detection. Exclusion boxes affect
references, not source identity; a redacted page gets a separate OCR cache entry.
Mistral calls are paid. Only detected reference spans (or manually selected pages)
are passed to bibliography extraction.

OpenAlex uses up to four workers, with a shared five-requests-per-second limiter,
including retries. Known DOIs are fetched in batches of 50. A DOI result still
has to pass metadata matching; successful matches skip title search. Missing or
conflicting DOI results fall back to title search. SQLite caches successful search
responses for seven days, DOI lookup responses for thirty days, and empty results
for one day; HTTP failures are not cached. This reduces repeated charges on reruns
without changing the matching thresholds. Cached data is re-evaluated by the current
matcher. Cache hits, DOI shortcuts, and fallback stages appear in the logs.

OpenAlex receives citation titles and supplied DOIs. It searches candidates and
compares titles, author surnames, publication years, and identifiers. Close or
conflicting candidates stay unresolved; matches are heuristic, not guarantees.
Accepted matches supply export metadata and DOI values; the original citation
and complete matching evidence remain available in the results and database.
API failures are recorded as errors rather than missing matches. See the
[OpenAlex API reference](https://help.openalex.org/api/llm-quick-reference/) and
[Mistral structured output documentation](https://docs.mistral.ai/inference/quickstarts/get-structured-output).

SQLite stores `papers`, `documents` (PDF SHA-256 and extracted source metadata),
and `citations` (directed source/target IDs, original citation, matching evidence).
Accepted OpenAlex work IDs unify papers across PDFs. Unresolved references have
separate identities, so similar titles cannot silently merge unrelated works.
Reprocessing the same PDF replaces its citation rows in a transaction. The browser's
**View saved citation graph** link (`/api/graph`) returns the persistent graph as JSON,
including match statuses. PDFs themselves remain temporary in-memory uploads.

Command-line analysis is also available:

```bash
uv run python mistral.py paper.pdf --database papers.sqlite3 --output bibliography.json
uv run python mistral.py paper.pdf --pages 12-15
```

The CLI writes JSON (including original citations and evidence), BibTeX and RIS.
Automatic section detection and model extraction can miss references; inspect the
results and use manual pages when necessary. Metadata matching deliberately leaves
incomplete or ambiguous citations unresolved.

Run offline regression tests:

```bash
uv run python -m unittest discover -s tests -v
```

The Mistral app logs extraction stages, OpenAlex HTTP status/retries, and each
citation's match decision/reason to the terminal. It does not log API keys or PDF
payloads. Set `BIB_LOG_LEVEL=WARNING` to reduce output. The browser shows estimated
stage progress and plain-language rejection reasons beside citations. Progress
percentages indicate stages/items, not predicted remaining time.

## Review matches and previous runs

**Previous runs & review matches** opens persistent run history. Each completed
extraction is saved separately, including repeated extractions of the same PDF.
Older results created before run history existed are imported as one snapshot per
PDF; their original timestamps and overwritten runs cannot be recovered.

In live results or a saved run, expand **Compare and select a match** to see saved
OpenAlex candidates with author, year, venue, DOI, and matching evidence. Open the
record to inspect it, choose **Use this match**, confirm an automatic match, or
leave the reference unresolved. The source paper identity can also be reviewed.
Selections immediately update that run's metadata, BibTeX/RIS exports and counts.
Automatic matches and your confirmations are counted separately; these are matching
statuses, not guarantees of bibliographic correctness. History can be filtered to
references that still need a decision. A completed run and its choices survive
server restarts; PDF previews still expire.

The graph represents the latest extraction per PDF. Changing that run updates its
graph links; reviewing an older run updates only that snapshot. Re-extracting creates
a new run with fresh automated decisions, preserving older manual choices in history.

BibTeX uses `@article` and `journal` for journal articles, `@book` for books,
`@incollection` and `booktitle` for chapters, and `@inproceedings` for conference
papers. A chapter's OpenAlex parent work supplies the book title when available;
a book series or repository is not substituted for it. Unknown publication types
remain `@misc`. Author order, year, volume, issue (`number`), page ranges (`--`), DOI,
and known publisher metadata are mapped into their respective fields. Missing
metadata is not invented, and the original printed citation remains available.

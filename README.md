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

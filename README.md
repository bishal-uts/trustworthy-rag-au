# Trustworthy RAG for Australian Financial Regulation

A calibrated three-state RAG (`confident` / `hedged` / `refused`) over Australian financial regulation, running on local-only inference. Hybrid retrieval (BM25 + dense + RRF), heuristic confidence routing, NLI-based faithfulness check, structured JSON output with citations.

## Run

```bash
# Install
python -m venv .venv
.venv\Scripts\activate                # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Start Ollama in another shell
ollama serve

# Pull models
python scripts/pull_models.py --essentials   # Ollama LLMs
python scripts/pull_models.py --hf-warm      # HuggingFace embedder + NLI

# Get docs, parse, chunk, index
python -m data.download --all
python -m src.parsing --all
python -m src.chunking --all
python scripts/build_index.py

# Smoke test
python scripts/smoke_test.py

# UI
python -m ui.gradio_app                       # http://127.0.0.1:7860
```

## Licence

MIT — see [LICENSE](LICENSE).

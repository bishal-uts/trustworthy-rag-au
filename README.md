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

## Evaluation

This repo ships a comparison framework that pits the trustworthy pipeline against four baseline RAG variants on the same benchmark. The point: show that the trust mechanisms (confidence routing + NLI faithfulness check) trade coverage for accuracy and cut hallucinations.

### Baselines

| Baseline | Retrieval | Refuse on low confidence | NLI faithfulness check |
| --- | --- | --- | --- |
| `no_rag` | none — LLM only | no | no |
| `dense_only` | FAISS dense | no | no |
| `hybrid` | BM25 + dense + RRF | no | no |
| `hybrid_refuse` | BM25 + dense + RRF | **yes** | no |
| `trustworthy` (this repo) | BM25 + dense + RRF | **yes** | **yes** |

Each row in the comparison table isolates the effect of one component.

### Metrics

| Metric | Definition |
| --- | --- |
| Coverage | fraction of questions the system answered (didn't refuse) |
| Accuracy on answered | fraction of answered questions that are correct |
| Effective accuracy | correct / total (combines coverage and accuracy) |
| Hallucination rate | fraction of answered questions with at least one unfaithful claim |
| Refusal precision | of refused outputs, fraction whose expected state was `refused` |
| Risk-adjusted loss | total loss under a configurable cost matrix (default: `wrong_confident=10`, `refused_answerable=1`) |

Correctness labels (`YES` / `PARTIAL` / `NO`) come from a composite judge that tries keyword match first, then NLI bidirectional entailment, then LLM-as-judge. Pass `--judge-llm <other_id>` to use a different LLM than the one under test (avoids self-favourability bias).

### Run

```bash
# All 5 baselines, default models
python -m eval.comparison_eval

# Subset of baselines, with a separate judge LLM
python -m eval.comparison_eval --baselines hybrid,trustworthy --judge-llm deepseek-r1-8b

# Per-category breakdown
python -m eval.comparison_eval --per-category

# Plot results
pip install matplotlib
python -m eval.plot_results eval/results/comparison_<timestamp>.json --per-category
```

Results land in `eval/results/comparison_<timestamp>.json`. A sample output with illustrative numbers is checked in at `eval/results/sample_output.json`, with rendered plots in `eval/results/sample_output_plots/`.

### Retrieval-only eval

When you want to compare embedders without paying LLM cost:

```bash
python -m eval.retrieval_eval --all-embedders --k 10
```

Computes Recall@k, Hit@k, MRR per embedder. Build the dense index for each embedder first (`python scripts/build_index.py --all-embedders`).

### Benchmark

`eval/benchmark.yaml` ships 20 questions across 7 categories (`answerable_easy`, `answerable_multi_chunk`, `answerable_crossdoc`, `answerable_inferential`, `unanswerable_out_of_scope`, `ambiguous`, `adversarial_misleading`). Target for full results: ~100 questions. Each question carries `expected_answer`, `expected_answer_keywords` (for cheap keyword judging), `expected_citations`, and `verified: true|false`.

## Licence

MIT — see [LICENSE](LICENSE). Note: `eval/benchmark.yaml` is **not** under MIT and stays private to the project.

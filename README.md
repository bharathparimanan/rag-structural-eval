# rag-structural-eval

**Controlled evaluation of structural retrieval interventions on Naive RAG pipelines —
comparing fixed-window, paragraph-boundary, and graph-constrained chunking strategies
on QASPER using RAGAS metrics.**

---

## Project Overview

This project is an MSc Data Science dissertation at the University of Hertfordshire
(Module 7PAM2002, Semester B 2025–26). It investigates whether structural constraints
derived from document topology can mitigate two well-documented failure modes in
Naive RAG pipelines: **Context Loss** and **Retrieval Noise**.

Three pipeline variants are compared on the QASPER scientific QA benchmark:

| Variant | Description |
|---------|-------------|
| **V0 — Naive RAG Baseline** | Fixed 256-token window chunking, FAISS top-K=5 cosine retrieval, no structural awareness |
| **V1 — Structure-Aware Chunking** | Paragraph-boundary chunking using QASPER JSON metadata, same retrieval as V0 |
| **V2 — Graph-Constrained Retrieval** | V1 chunking + NetworkX document graph with 2-hop proximity filter, adjacency expansion, and Provenance Coverage metric |

### Research Question

> To what extent do graph-based structural constraints on vector retrieval mitigate
> Context Loss and Retrieval Noise in Naive RAG pipelines, as measured by RAGAS
> Context Recall, Context Precision, Faithfulness, and Provenance Coverage
> on the QASPER dataset?

---

## Repository Structure

```
rag-structural-eval/
├── pipeline/
│   ├── source/                    # Dataset identity and access config
│   ├── ingest/                    # Pull QASPER from HuggingFace Hub
│   ├── raw/                       # Immutable local QASPER data (gitignored)
│   ├── transform/
│   │   ├── shared/                # Embedder (all-MiniLM-L6-v2) + query sampler
│   │   ├── v0/                    # Fixed 256-token chunker
│   │   ├── v1/                    # Paragraph-boundary chunker
│   │   └── v2/                    # Same as V1 (V2 diverges at model layer)
│   ├── model/
│   │   ├── shared/                # FAISS store + Llama 3.2 3B generator
│   │   ├── v0/                    # Top-K=5 cosine retriever
│   │   ├── v1/                    # Top-K=5 cosine retriever (paragraph chunks)
│   │   └── v2/                    # Graph builder + graph-constrained retriever
│   └── serve/
│       ├── shared/                # RAGAS evaluation (all variants)
│       ├── v0/                    # V0 evaluation orchestrator
│       ├── v1/                    # V1 evaluation orchestrator
│       └── v2/                    # V2 evaluation orchestrator + provenance metric
├── runners/
│   ├── run_v0.py                  # End-to-end V0 pipeline entry point
│   ├── run_v1.py                  # End-to-end V1 pipeline entry point
│   └── run_v2.py                  # End-to-end V2 pipeline entry point
├── notebooks/
│   ├── 00_qasper_inspection.ipynb # QASPER schema and field exploration
│   ├── 01_eda_corpus_stats.ipynb  # Corpus statistics and distributions
│   └── 02_embedding_sanity.ipynb  # Embedding quality spot-check
├── artifacts/
│   ├── embeddings/                # Saved .npy embeddings per variant (gitignored)
│   ├── indexes/                   # Serialised FAISS indexes per variant (gitignored)
│   └── results/
│       ├── v0/                    # results.csv (gitignored) + metrics_summary.json
│       ├── v1/                    # results.csv (gitignored) + metrics_summary.json
│       └── v2/                    # results.csv (gitignored) + metrics_summary.json
├── docs/
│   ├── v0_development_clarification.docx
│   └── repo_architecture.docx
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.11 |
| Ollama | Latest | https://ollama.com — must be running before pipeline executes |
| Llama 3.2 3B | 3B (4-bit) | Pulled via Ollama — ~2GB download |
| RAM | 8GB minimum | M1 Mac 8GB unified memory — pipeline runs sequentially to avoid OOM |

---

## Installation

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/bharathparimanan/rag-structural-eval.git
cd rag-structural-eval
```

**Step 2 — Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Install and start Ollama:**
```bash
brew install ollama
ollama serve &
ollama pull llama3.2:3b
```

**Step 5 — Verify Ollama is working:**
```bash
ollama run llama3.2:3b "Say: OLLAMA OK"
```

---

## Running the Pipeline

### First Run (full setup)

```bash
# 1. Download QASPER from HuggingFace Hub (run once — saves to pipeline/raw/)
python pipeline/ingest/download.py

# 2. Validate downloaded data
python pipeline/ingest/validate.py

# 3. Draw and lock the stratified 150-query sample (run once)
python pipeline/transform/shared/query_sampler.py

# 4. Run V0 end-to-end
python runners/run_v0.py
```

### Subsequent Runs (skip re-embedding)

```bash
# Reuse saved FAISS index and embeddings
python runners/run_v0.py --skip-embed
```

### Running V1 and V2

```bash
python runners/run_v1.py
python runners/run_v1.py --skip-embed

python runners/run_v2.py
python runners/run_v2.py --skip-embed
```

---

## Results

After each run, results are written to `artifacts/results/<variant>/`:

- `metrics_summary.json` — mean and standard deviation for each RAGAS metric across all queries *(committed to GitHub)*
- `results.csv` — per-query scores: context_recall, context_precision, faithfulness, wall_clock_seconds *(gitignored — regenerate locally)*

### Baseline Results (V0)

> Populated after first full evaluation run.

| Metric | V0 Mean | V0 Std |
|--------|---------|--------|
| Context Recall | — | — |
| Context Precision | — | — |
| Faithfulness | — | — |

---

## Technical Stack

| Component | Choice | Version | Justification |
|-----------|--------|---------|---------------|
| Dataset | QASPER (allenai/qasper) | — | Ground-truth evidence annotations enable RAGAS Context Recall; structured JSON provides paragraph boundaries for V1/V2 |
| Embedding model | all-MiniLM-L6-v2 | 384-dim | Benchmarked on BEIR/MTEB; runs on CPU/MPS; MIT licence; full corpus fits in <100MB RAM |
| Vector index | FAISS IndexFlatL2 | faiss-cpu | Exact search — deterministic, reproducible, no approximation noise |
| LLM | Llama 3.2 3B | via Ollama | Runs locally on M1 8GB at ~2.5GB; zero API cost; temperature=0 for reproducibility |
| Graph library | NetworkX | — | In-memory; no infrastructure overhead; built-in BFS/shortest-path for V2 hop filter |
| Evaluation | RAGAS | — | Component-level metrics separating retrieval and generation failures |

---

## Dataset

**QASPER** — Question Answering over Scientific Papers with Evidence
- Published by: Allen Institute for Artificial Intelligence (AllenAI)
- Source: [allenai/qasper on HuggingFace](https://huggingface.co/datasets/allenai/qasper)
- Licence: **CC BY 4.0** — free for research use with attribution
- Paper: Dasigi et al. (2021), *A Dataset for Information-Seeking Questions and Answers in Research Papers*, NAACL 2021
- Contains no personal data — published academic NLP papers only
- No UH Ethics Committee approval required

---

## Evaluation Metrics

| Metric | Failure Mode Targeted | Threshold for Problem |
|--------|-----------------------|-----------------------|
| RAGAS Context Recall | Context Loss | < 0.70 |
| RAGAS Context Precision | Retrieval Noise | < 0.70 |
| RAGAS Faithfulness | Generation impact of Retrieval Noise | < 0.75 |
| Provenance Coverage | Explainability (V2 only) | Descriptive — no threshold |

---

## Architecture Documentation

Full design decisions, DE layer mapping, and pre-implementation agreements are documented in:

- `docs/repo_architecture.docx` — repository structure and pipeline design
- `docs/v0_development_clarification.docx` — V0 scope lock, technical stack justification, output contract

---

## Licence

This project is licensed under the **MIT Licence** — see [LICENSE](LICENSE) for details.

Dataset (QASPER) is licensed under **CC BY 4.0** by the Allen Institute for AI.
Embedding model (all-MiniLM-L6-v2) is licensed under **MIT** by Microsoft.

---

## Author

**Bharath Parimanan**
MSc Data Science — University of Hertfordshire
Supervisor: Hasan Al-Madfai
# Graph-Constrained Vector Retrieval for Mitigating Structural Failure Modes in Naive RAG Pipelines

**University of Hertfordshire — MSc Data Science (7PAM2002)**  
**Author:** Bharath Parimanan (SRN: 24064964)  
**Supervisors:** Dr. Hasan Al-Madfai  
**GitHub:** https://github.com/bharathparimanan/rag-structural-eval

---

## What This Project Does

Naive RAG pipelines fail in two structurally predictable ways:

- **Context Loss** — fixed-window chunking severs evidence at arbitrary token boundaries, splitting coherent arguments across non-adjacent chunks
- **Retrieval Noise** — cosine similarity retrieves topically similar but evidentially irrelevant chunks into the LLM context

This project builds and evaluates three pipeline variants that isolate the independent contribution of each fix — one change at a time, everything else held constant.

---

## Research Question

> To what extent do graph-structural constraints over document paragraph representations mitigate the context loss and retrieval noise failure modes of naive RAG pipelines, as measured by RAGAS evaluation metrics on a scientific question-answering corpus?

---

## Three-Variant Controlled Design

| Variant | What Changes | What Is Fixed | Purpose |
|---------|-------------|---------------|---------|
| **V0** Naive RAG | Fixed-window chunking (256 tokens, 32 overlap) + cosine similarity Top-5 | Everything else | Performance floor — the failure condition |
| **V1** Structure-Aware | Paragraph-boundary chunking + cosine similarity Top-5 | Everything else | Isolates chunking effect (V0 vs V1) |
| **V2** Graph-Constrained | Paragraph-boundary chunking + graph proximity re-ranking Top-10→5 | Everything else | Isolates graph constraint effect (V1 vs V2) |

- **V0 vs V1** → chunking effect alone
- **V1 vs V2** → graph constraint effect alone
- **V0 vs V2** → combined effect of both interventions

---

## Technical Stack

| Component | Choice |
|-----------|--------|
| Dataset | QASPER (allenai/qasper, train split) |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Vector index | FAISS IndexFlatL2 |
| Language model | Llama 3.1 8B via Groq API |
| Graph library | NetworkX (~46,882 nodes, ~81,346 edges) |
| Evaluation | RAGAS (Context Recall, Context Precision, Faithfulness, Answer Relevancy) + Provenance Coverage |

---

## Repository Structure

```
rag-structural-eval/
├── notebooks/
│   ├── 00_poc_single_document.ipynb          ← proof of concept
│   ├── v0_naive_rag.ipynb                    ← V0 baseline pipeline
│   ├── v1_structure_aware_chunking.ipynb     ← V1 paragraph chunking
│   ├── v2_graph_constrained_retrieval.ipynb  ← V2 graph constraint
│   ├── full_evaluation_groq.ipynb            ← full 50-query evaluation
│   └── experimental_analysis_and_visualisation.ipynb
│   └── data/                                 ← pre-built artefacts
│       ├── queries.json
│       ├── v0_chunks.json
│       ├── v0_index.faiss
│       ├── v1_chunks.json
│       ├── v1_index.faiss
│       ├── v2_graph.pkl
│       └── full_eval_summary.json
├── visuals/
├── .env.example                              ← API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Reproduce

**1. Clone the repository**

```bash
git clone https://github.com/bharathparimanan/rag-structural-eval
cd rag-structural-eval
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set up your Groq API key**

```bash
cp .env.example .env
# Open .env and add your Groq API key
# Get a free key at https://console.groq.com (no credit card required)
```

**4. Run the full evaluation**

Open `notebooks/full_evaluation_groq.ipynb` and run all cells sequentially.  
Pre-built artefacts in `notebooks/data/` load automatically — no rebuild required.

The notebook runs V0 → V1 → V2 sequentially against 50 stratified queries.  
Results are saved incrementally after every query — if interrupted, re-run and it resumes from where it stopped.

---

## V2 Graph Parameters

```
HOP_LIMIT       = 2      # 2-hop neighbourhood around anchor node
PROXIMITY_BONUS = +0.3   # score bonus for chunks within HOP_LIMIT
PENALTY         = x0.5   # score multiplier for chunks outside HOP_LIMIT
K_INITIAL       = 10     # widened FAISS retrieval before graph filter
K_FINAL         = 5      # chunks passed to LLM after re-ranking
```

Parameters selected via 30-query ablation study testing HOP_LIMIT ∈ {1,2,3}, PROXIMITY_BONUS ∈ {0.1,0.3,0.5}, and PENALTY ∈ {×0.25,×0.5,×0.75}.

---

## Pilot Results (n=10 queries)

| Metric | V0 Naive RAG | V1 Structure-Aware | V2 Graph-Constrained |
|--------|-------------|-------------------|---------------------|
| Context Recall | 0.475 | 0.436 | **0.498** |
| Provenance Coverage | N/A | N/A | 0.4776 |

Context Precision and Faithfulness scores from the pilot are unreliable due to Llama 3.2 3B JSON parse failures under RAGAS. The full evaluation uses Llama 3.1 8B via Groq which resolves the structured output issue.

---

## Ethical Compliance

QASPER contains no personal data. It consists entirely of published NLP research papers available under CC BY 4.0. No human participants were involved. No UH Ethics Committee approval is required.

---

## Key References

- Dasigi et al. (2021) — QASPER dataset, NAACL 2021
- Es et al. (2024) — RAGAS evaluation framework, EACL 2024
- Lewis et al. (2020) — Original RAG paper, NeurIPS 2020
- Wang et al. (2023) — Knowledge Graph Prompting, AAAI 2023
- Edge et al. (2024) — GraphRAG, arXiv preprint
- Gao et al. (2023) — RAG survey, arXiv preprint

Full reference list in Harvard format in the Final Project Report.

---

*MSc Data Science — University of Hertfordshire — Module 7PAM2002 — Semester B 2025–26*
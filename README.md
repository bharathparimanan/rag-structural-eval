# Graph-Constrained Vector Retrieval for Mitigating Structural Failure Modes in Naive RAG Pipelines

**University of Hertfordshire - MSc Data Science (7PAM2002)**  
**Author:** Bharath Parimanan (SRN: 24064964)  
**Supervisors:** Dr. Hasan Al-Madfai · Dr. Vid Irsic  
**GitHub:** https://github.com/bharathparimanan/rag-structural-eval

---

## What This Project Does

This project builds and evaluates three RAG pipeline variants to isolate
the independent contribution of graph-structural constraints on retrieval
quality. Two structural failure modes of naive RAG are targeted:

- **Context Loss** - fixed-window chunking severs evidence at arbitrary token boundaries
- **Retrieval Noise** - cosine similarity retrieves topically similar but evidentially irrelevant chunks

---

## Three Pipeline Variants

| Variant | Chunking | Retrieval | Purpose |
|---------|----------|-----------|---------|
| **V0** Naive RAG | Fixed window (256 tokens, 32 overlap) | Cosine similarity Top-5 | Baseline |
| **V1** Structure-Aware | Paragraph boundary | Cosine similarity Top-5 | Isolates chunking effect |
| **V2** Graph-Constrained | Paragraph boundary | Graph proximity re-ranking Top-10 to 5 | Isolates graph constraint effect |

- V0 vs V1 - chunking effect alone
- V1 vs V2 - graph constraint effect alone
- V0 vs V2 - combined effect

---

## Quick Start

**1. Clone**
```bash
git clone https://github.com/bharathparimanan/rag-structural-eval
cd rag-structural-eval
```

**2. Install**
```bash
pip install -r requirements.txt
```

**3. Set API key**
```bash
cp .env.example .env
# Add your free Groq API key - get one at https://console.groq.com
```

**4. Run evaluation**

Open `notebooks/full_evaluation.ipynb` and run all cells sequentially.  
Pre-built artefacts in `notebooks/data/` load automatically.  
Results save incrementally after every query - safe to interrupt and resume.

---

## Repository Structure

```
rag-structural-eval/
├── notebooks/
│   ├── 00_poc_single_document.ipynb          <- proof of concept
│   ├── v0_naive_rag.ipynb                    <- V0 baseline pipeline
│   ├── v1_structure_aware_chunking.ipynb     <- V1 paragraph chunking
│   ├── v2_graph_constrained_retrieval.ipynb  <- V2 graph constraint
│   ├── full_evaluation.ipynb                 <- full evaluation run
│   ├── experimental_analysis_and_visualisation.ipynb
│   └── data/                                 <- pre-built artefacts
│       ├── queries.json                      <- 50 stratified queries (seed=42)
│       ├── v0_chunks.json / v0_index.faiss
│       ├── v1_chunks.json / v1_index.faiss
│       └── v2_graph.pkl
├── docs/
│   ├── methodology_notes.md                  <- design rationale and parameter selection
│   └── known_issues.md                       <- issue log with root causes and fixes
├── results/
│   ├── pilot_results.md                      <- 10-query pilot summary
│   └── full_eval_summary.json                <- full evaluation results
├── visuals/                                  <- pipeline and graph diagrams
├── .env.example                              <- API key template
├── requirements.txt
└── README.md
```

---

## Stack

| Component | Choice |
|-----------|--------|
| Dataset | QASPER (allenai/qasper, train split) |
| Embedding | all-MiniLM-L6-v2 (384-dim) |
| Vector index | FAISS IndexFlatL2 (exact search) |
| Language model | Llama 3.1 8B via Groq API |
| Graph | NetworkX (~46,882 nodes, ~81,346 edges) |
| Evaluation | RAGAS (4 metrics) + Provenance Coverage |

---

## V2 Graph Parameters

Selected via 30-query ablation study:

```
HOP_LIMIT       = 2      # neighbourhood depth around anchor node
PROXIMITY_BONUS = +0.3   # score boost for chunks within HOP_LIMIT
PENALTY         = x0.5   # score reduction for chunks outside HOP_LIMIT
K_INITIAL       = 10     # widened FAISS retrieval before graph filter
K_FINAL         = 5      # chunks passed to LLM after re-ranking
```

---

## Pilot Results (n=10 queries)

| Metric | V0 Naive RAG | V1 Structure-Aware | V2 Graph-Constrained |
|--------|-------------|-------------------|---------------------|
| Context Recall | 0.475 | 0.436 | **0.498** |
| Provenance Coverage | N/A | N/A | 0.480 |

Full evaluation: 50 queries x 3 variants = 150 controlled evaluation instances.  
Results in `results/full_eval_summary.json`.

---

## Reproducibility Notes

- All random operations use seed=42
- Temperature set to 0 for deterministic generation
- RAGAS requires `.to_pandas()` not `.get()` in updated versions
- Set `SLEEP_BETWEEN_CALLS = 12` for Groq free tier rate limiting
- Resume logic handles interruptions - re-run notebook to continue

---

## Ethics

QASPER contains no personal data. Published NLP papers under CC BY 4.0.  
No human participants. No UH Ethics Committee approval required.

# Graph-Constrained Vector Retrieval for Mitigating Structural Failure Modes in Naive RAG Pipelines

**University of Hertfordshire — MSc Data Science (7PAM2002)**  
**Author:** Bharath Parimanan  
**Supervisors:** Dr. Vid Irsic · Hasan Al-Madfai  
**GitHub:** [github.com/bharathparimanan/rag-structural-eval](https://github.com/bharathparimanan/rag-structural-eval)

---

## What Problem Does This Solve?

Naive RAG pipelines fail in two structurally predictable ways: fixed-window chunking destroys document topology, causing evidence to be split and lost (Context Loss); and pure cosine similarity retrieval floods the LLM with plausible but irrelevant context (Retrieval Noise). This project introduces a graph-constrained retrieval layer built on NetworkX to address both failure modes in a controlled, measurable experiment — without modifying the embedding model, the LLM, or the vector index.

---

## Research Question

> *To what extent do graph-based structural constraints on vector retrieval mitigate Context Loss and Retrieval Noise in Naive RAG pipelines, as measured by RAGAS Context Recall, Context Precision, Faithfulness, and Provenance Coverage on the QASPER dataset?*

---

## Pipeline Architecture

```
QASPER Dataset (AllenAI)
        │
        ▼
┌─────────────────┐
│  data/           │  ← Single entry point: loads QASPER, draws 150-query stratified sample
│  load_qasper.py  │     Same query set hits all three variants — eliminates query confounds
└────────┬────────┘
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌────────────────────┐                   ┌─────────────────────┐
│  graph/             │                   │  assets/indexes/     │
│  build_graph.py     │                   │  faiss_v0/v1/v2      │
│  → qasper_graph.pkl │                   │  .index              │
└────────┬───────────┘                   └──────────┬──────────┘
         │  Built once. Loaded by all variants.      │
         └──────────────────┬───────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ V0 Naive │  │ V1 Struct│  │ V2 Graph │
       │ pipelines│  │ pipelines│  │ pipelines│
       │ /v0_.py  │  │ /v1_.py  │  │ /v2_.py  │
       └────┬─────┘  └────┬─────┘  └────┬─────┘
            │              │              │
            └──────────────┴──────────────┘
                           │
                           ▼
                ┌──────────────────┐
                │  evaluation/      │
                │  run_eval.py      │  ← RAGAS runner decoupled from pipeline logic
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  assets/results/  │
                │  full_eval_       │
                │  results.json     │
                └──────────────────┘
```

---

## Three-Variant Controlled Experiment

| Variant | Independent Variable | Fixed | Measures |
|---------|---------------------|-------|---------|
| **V0** Naive RAG | Fixed 256-token window chunking, K=5 | Embedding, LLM, FAISS, query set | Baseline Context Loss + Retrieval Noise |
| **V1** Structural | Paragraph-boundary chunking, K=5 | Everything else | Chunking effect in isolation (SQ1) |
| **V2** Graph-Constrained | Graph proximity filter + adjacency expansion, K=10→5 | Everything else | Graph constraint contribution (SQ2, SQ3) |

**Comparison logic:**
- V0 vs V1 → isolates chunking effect
- V1 vs V2 → isolates graph constraint contribution
- V0 vs V2 → total combined effect of the full structural intervention

---

## Failure Modes and Where They Live

```
Chunking Step         →  Context Loss     →  fixed windows split evidence across non-adjacent chunks
Retrieval Step        →  Retrieval Noise  →  cosine similarity ranks topically similar but
                                              evidentially irrelevant chunks into top-K
```

**V2 injection point:** Post-FAISS graph proximity filter re-scores K=10 candidates by structural distance to anchor node, then adjacency expansion pulls in neighbours FAISS missed. The graph does not replace FAISS — it constrains its output.

---

## Evaluation Metrics

| Metric | Failure Mode Targeted | Threshold | Framework |
|--------|----------------------|-----------|-----------|
| Context Recall | Context Loss | < 0.70 = problem | RAGAS |
| Context Precision | Retrieval Noise | < 0.70 = problem | RAGAS |
| Faithfulness | Generation impact of Retrieval Noise | < 0.75 = problem | RAGAS |
| Provenance Coverage | Explainability (V2 only) | Descriptive | Custom — graph path computation |

No single-metric evaluation. Retrieval failures are measured independently from generation failures.

---

## Repository Structure

```
rag-structural-eval/
│
├── pipelines/        # Three isolated variant implementations — change one variable,
│                     # nothing else moves
│
├── graph/            # Graph construction and serialisation — built once,
│                     # loaded by all variants, never rebuilt mid-experiment
│
├── evaluation/       # RAGAS runner decoupled from pipeline logic — retrieval
│                     # errors and generation errors measured independently
│
├── data/             # Single entry point for QASPER loading and 150-query
│                     # stratified sampling — same query set hits all three variants
│
├── notebooks/        # Exploratory and validation layer — each variant prototyped
│                     # in isolation before logic was promoted to .py modules
│
├── assets/           # Serialised artifacts (graph .pkl, FAISS indexes, results
│                     # .json) — built once, reused across runs, never regenerated unnecessarily
│
├── docs/             # Architecture, metric definitions, and ablation notes —
│                     # satisfies handbook README and code documentation requirement
│
├── tests/            # Unit tests on chunking logic and graph construction —
│                     # validates controlled variables behave as specified
│
├── config.py         # Single source of truth for K, HOP_LIMIT, PROXIMITY_BONUS,
│                     # PENALTY — guarantees parameter consistency across all three variants
│
├── main.py           # Single entry point that runs all three variants sequentially
│                     # and writes results — one command reproduces the full experiment
│
├── requirements.txt  # Pinned dependency versions — ensures the experiment is
│                     # reproducible on any machine including university HPC
│
└── README.md         # Answers what, why, how, and what next —
                      # primary artifact a marker or examiner reads first
```

---

## Technical Stack

| Component | Choice | Justification |
|-----------|--------|---------------|
| Dataset | QASPER (AllenAI, CC BY 4.0) | Ground-truth evidence paragraphs enable Context Recall computation; structured JSON exposes paragraph boundaries for graph construction |
| Embedding | all-MiniLM-L6-v2 (384-dim) | Validated on BEIR/MTEB retrieval benchmarks; compact enough to store full QASPER corpus in under 100MB RAM; held constant across all variants |
| LLM | Llama 3.1 8B via Ollama (temp=0) | Local inference — zero API cost, reproducible; temperature=0 eliminates non-determinism across evaluation runs |
| Vector Index | FAISS IndexFlatL2 | Exact search — no approximation error; variations between V0/V1/V2 attributable solely to experimental intervention |
| Graph | NetworkX (~59K nodes, ~81K edges) | In-memory; zero infrastructure overhead; BFS/shortest-path built-in; serialised as .pkl for reuse |
| Evaluation | RAGAS (Es et al., 2023) | Component-level metrics separate retrieval failures from generation failures — standard QA metrics cannot do this |
| HPC | University cluster V100/A100 via SLURM | Full 150-query evaluation across 3 variants; SLURM job scripts in `docs/` |

---

## Graph Construction Rules (V2)

- **Paragraph nodes:** Each QASPER paragraph = one node (paper ID, section title, paragraph index, text)
- **Edge Type 1 — Sequential adjacency:** Undirected edge between paragraph N and N+1 within the same section — encodes local narrative continuity
- **Edge Type 2 — Section membership:** All paragraphs in the same section connect to a virtual section node — encodes hierarchical co-membership
- **No inter-paper edges:** Graph is a forest of paper sub-graphs; cross-paper edges are not constructed

**V2 Retrieval Parameters:**

```python
K               = 10      # widened initial retrieval for graph filter candidates
HOP_LIMIT       = 2       # maximum graph distance from anchor to receive bonus
PROXIMITY_BONUS = +0.3    # added to cosine score for chunks within HOP_LIMIT
PENALTY         = x0.5    # multiplied against cosine score for disconnected chunks
FINAL_TOP_K     = 5       # final chunks passed to LLM after graph re-scoring
```

All parameters are defined once in `config.py` and imported by all pipeline variants.

---

## How to Reproduce

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the graph (once)
```bash
python graph/build_graph.py
# Output: assets/graphs/qasper_graph.pkl
```

### 3. Run full experiment
```bash
python main.py
# Runs V0 → V1 → V2 sequentially
# Output: assets/results/full_eval_results.json
```

### 4. Run ablation study (optional)
```bash
python evaluation/run_ablation.py
# Sweeps HOP_LIMIT ∈ {1,2,3}, PROXIMITY_BONUS ∈ {0.1,0.3,0.5}, K ∈ {8,10,12}
# Output: assets/results/ablation_results.json
```

### 5. Explore notebooks
```
notebooks/00_poc_sample_pdf.ipynb    ← feasibility proof on single document
notebooks/01_pilot_v0.ipynb          ← V0 baseline (10-query pilot)
notebooks/02_pilot_v1.ipynb          ← V1 chunking intervention
notebooks/03_pilot_v2.ipynb          ← V2 full structural intervention
notebooks/04_results_analysis.ipynb  ← cross-variant comparison and plots
```

---

## Ablation Studies

| Parameter | Values Tested | Fixed During Run |
|-----------|--------------|-----------------|
| HOP_LIMIT | 1, 2, 3 | PROXIMITY_BONUS=0.3, K=10 |
| PROXIMITY_BONUS | 0.1, 0.3, 0.5 | HOP_LIMIT=2, K=10 |
| K (initial retrieval) | 8, 10, 12 | HOP_LIMIT=2, PROXIMITY_BONUS=0.3 |

Evaluated on a 30-query subset. Wilcoxon signed-rank test used for statistical validity on the full 150-query results.

---

## Pilot Results (n=10 queries)

| Metric | V0 Naive | V1 Structural | V2 Graph |
|--------|----------|---------------|----------|
| Context Recall | 0.475 | 0.436 | **0.498** |
| Provenance Coverage | — | — | 0.48 |

*Context Precision and Faithfulness scores from pilot are unreliable — Llama 3.2 3B JSON output failures in RAGAS structured evaluation. Full results use Llama 3.1 8B on HPC.*

---

## Ethical Compliance

QASPER contains no personal data. It consists entirely of published NLP research papers licensed under CC BY 4.0. No UH Ethics Committee approval is required. No surveys, interviews, or human participants are involved. All data is secondary and publicly available.

**Licence:** CC BY 4.0 — Dasigi et al. (2021), NAACL 2021.

---

## Key References

- Dasigi et al. (2021) — QASPER dataset, NAACL 2021
- Es et al. (2023) — RAGAS evaluation framework, EACL 2024
- Gao et al. (2023) — RAG survey
- Edge et al. (2024) — GraphRAG
- Peng et al. (2024) — Graph-enhanced RAG
- Wang et al. (2023) — RAG failure taxonomy

Full reference list in Harvard format in the Final Project Report.

---

## What Would Change at Scale

- Replace NetworkX with Neo4j for graphs exceeding available RAM
- Replace FAISS IndexFlatL2 with HNSW for sub-linear retrieval at corpus sizes above 1M chunks
- Replace Ollama local inference with vLLM or TGI for batch throughput on GPU cluster
- Add streaming ingestion layer (Kafka) if corpus updates are continuous rather than batch

---

*MSc Data Science — University of Hertfordshire — Module 7PAM2002 — Semester B 2025–26*
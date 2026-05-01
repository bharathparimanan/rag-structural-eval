# Pilot Evaluation Results

## Graph-Constrained Vector Retrieval - MSc Dissertation 7PAM2002

---

## Overview

Prior to the full 50-query evaluation, each pipeline variant was validated on a 10-query pilot study. The pilot was designed to confirm that all three pipeline variants were functioning correctly before scaling to the full evaluation run.

| Property | Detail |
|----------|--------|
| Sample size | 10 queries |
| Sampling strategy | Stratified - seed=42 |
| Inference model | Llama 3.2 3B via Ollama (local) |
| Evaluation framework | RAGAS |
| Dataset | QASPER train split |

---

## Query Type Distribution

| Answer Type | Count | Proportion |
|-------------|-------|------------|
| Extractive | 6 | 60% |
| Abstractive | 3 | 30% |
| Yes/No | 1 | 10% |

---

## Primary Results

| Metric | V0 Naive RAG | V1 Structure-Aware | V2 Graph-Constrained |
|--------|-------------|-------------------|---------------------|
| Context Recall | 0.475 | 0.436 | **0.498** |
| Context Precision | 0.270 | - | - |
| Faithfulness | 0.083 | - | - |
| Answer Relevancy | - | - | - |
| Provenance Coverage | N/A | N/A | 0.480 |

Dashes (-) indicate scores excluded as unreliable. See notes below.

---

## Delta Analysis

| Comparison | What It Isolates | Context Recall Change |
|------------|-----------------|----------------------|
| V0 → V1 | Chunking effect alone | −0.039 |
| V1 → V2 | Graph constraint effect alone | +0.062 |
| V0 → V2 | Combined effect | +0.023 |

---

## Key Findings

**V0 → V1 (Chunking Effect)**
Paragraph-boundary chunking produced a marginal decline in Context Recall
from 0.475 to 0.436. This counterintuitive result is attributed to sample
size - at n=10, a single query result in either direction shifts the mean
substantially. Jimeno Yepes et al. (2024) demonstrate element-based
chunking consistently outperforms fixed-window baselines at scale.
The pilot result is treated as directional only.

**V1 → V2 (Graph Constraint Effect)**
Graph-constrained retrieval improved Context Recall from 0.436 to 0.498
- a gain of 0.062. This is the most important finding from the pilot.
It demonstrates the graph proximity filter is recovering the recall loss
from V1 and improving beyond the V0 baseline.

**Provenance Coverage**
V2 Provenance Coverage of 0.480 confirms the graph constraint is
functioning as designed - approximately 48% of final retrieved chunks
have a traceable structural path to the anchor node within the 2-hop
neighbourhood. This metric is computed from graph metadata without LLM
involvement and is therefore not affected by the JSON parse issues
noted below.

---

## Reliability Notes

### Why V1 and V2 Context Precision and Faithfulness Are Excluded

Context Precision, Faithfulness, and Answer Relevancy require the
evaluation LLM to return structured JSON responses. Llama 3.2 3B via
Ollama consistently failed to produce valid JSON under the RAGAS
evaluation framework during the pilot, resulting in NaN scores handled
by the safe_score() wrapper as 0.0.

V0 scores for Context Precision (0.270) and Faithfulness (0.083) are
reported because a subset of V0 queries produced valid JSON responses
before the failure pattern became consistent. These scores should be
interpreted with caution.

### Why the Full Evaluation Uses a Different Model

The pilot failure motivated upgrading from Llama 3.2 3B (local, Ollama)
to Llama 3.1 8B (Groq API) for the full evaluation. The larger model
follows RAGAS structured output instructions reliably, resolving the
JSON parse failures.

### RAGAS API Compatibility Issue

During the full 50-query evaluation run, a separate RAGAS API
incompatibility was encountered - EvaluationResult no longer supports
.get() in updated RAGAS versions. The correct extraction method is
.to_pandas(). This fix was applied to full_evaluation_groq.ipynb before
the corrected full evaluation run. See docs/known_issues.md Issue 6
for full details.

---

## Pilot vs Full Evaluation Comparison

| Property | Pilot | Full Evaluation |
|----------|-------|-----------------|
| Queries per variant | 10 | 50 |
| Total evaluation instances | 30 | 150 |
| Inference model | Llama 3.2 3B (Ollama) | Llama 3.1 8B (Groq API) |
| RAGAS judge model | Llama 3.2 3B | Llama 3.1 8B |
| LLM-dependent metrics reliable | Partial (V0 only) | Yes (all variants) |
| Provenance Coverage | V2: 0.480 | V2: 0.4776 |

---

## Files

| File | Description |
|------|-------------|
| notebooks/data/v0_results.json | V0 pilot per-query results (10 queries) |
| notebooks/data/v1_results.json | V1 pilot per-query results (10 queries) |
| notebooks/data/v2_results.json | V2 pilot per-query results (10 queries) |
| results/full_eval_summary.json | Full evaluation aggregated results (50 queries x 3 variants) |

---

*Pilot conducted during iterative development phase.
Full evaluation results in full_eval_summary.json.*
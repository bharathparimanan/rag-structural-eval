# Methodology Notes

## Graph-Constrained Vector Retrieval for Mitigating Structural Failure Modes in Naive RAG

**Author:** Bharath Parimanan (SRN: 24064964)  
**Module:** 7PAM2002 - MSc Data Science, University of Hertfordshire  
**Supervisors:** Dr. Hasan Al-Madfai

---

## 1. Research Problem

Naive RAG pipelines exhibit two structural failure modes that are architectural in origin - meaning they cannot be corrected by improving the underlying language model.

### 1.1 Context Loss

Fixed-window chunking segments documents by token count without any awareness of document structure. A paragraph that contains a complete argument, evidence unit, or methodological step gets split at an arbitrary token boundary. The result is that the first half of the evidence lands in one chunk and the second half lands in the next. At retrieval time, cosine similarity may rank the first half in the top-K and leave the second half unranked, or vice versa. The LLM receives partial evidence and either generates an incomplete answer or confabulates the missing half from parametric memory.

The problem is not that the embedding model is weak - it is that the unit of text being embedded does not correspond to a meaningful semantic unit in the first place. No embedding model can recover meaning from a fragment.

### 1.2 Retrieval Noise

Cosine similarity between dense embedding vectors measures surface-level semantic proximity. This is not the same as evidential relevance. A chunk that shares vocabulary and topic with the query but does not contain the answer to the query will score highly. In practice this means the top-K retrieved set is contaminated with thematically related but evidentially irrelevant chunks that compete with the genuinely relevant passage for the LLM's attention window.

Steck et al. (2024) demonstrate that cosine similarity on learned embeddings can produce scores that are structurally arbitrary - driven by regularisation effects in the embedding model rather than genuine semantic proximity. This means the problem is not just that cosine similarity is a weak relevance signal. It is that the scores can be actively misleading.

---

## 2. Experimental Design Rationale

### 2.1 Why Three Variants

The existing literature on graph-based RAG - including GraphRAG (Edge et al., 2024) and Knowledge Graph Prompting (Wang et al., 2023) - demonstrates that structural approaches consistently outperform naive RAG. However, every published system that introduces graph structure simultaneously modifies multiple pipeline components. GraphRAG changes entity extraction, community detection, summary generation, and retrieval simultaneously. KGP changes graph construction, traversal strategy, and retrieval simultaneously. This means the observed performance gains cannot be attributed to any single component.

The three-variant design isolates each intervention:

- V0 establishes the failure condition with no structural awareness at any stage
- V1 adds structural awareness only at the chunking stage - everything else identical to V0
- V2 adds graph-structural awareness only at the retrieval stage - everything else identical to V1

This means any score difference between V0 and V1 is attributable to chunking strategy alone. Any score difference between V1 and V2 is attributable to the graph constraint alone. The design makes causal attribution possible, which no prior published study has enabled.

### 2.2 Why QASPER

Three properties of QASPER make it the right evaluation corpus for this study.

First, questions require evidence retrieval across the full body of a paper. Questions are written by practitioners who have read only the title and abstract - they do not know which section contains the answer. This means answering correctly requires the pipeline to locate content distributed across introduction, methodology, results, and discussion sections. This is precisely the condition under which fixed-window chunking is most disruptive.

Second, QASPER provides annotated supporting evidence passages alongside each answer. This makes it possible to compute Context Recall and Provenance Coverage directly - both require knowing which passages ground the correct answer. Without evidence annotation, measuring whether the right content was retrieved is not possible.

Third, NLP research papers follow a recognisable structure - abstract, introduction, related work, methodology, experiments, conclusion. This regularity means paragraph-boundary chunking is a meaningful structural intervention. The documents have genuine structure to preserve, and that structure is systematically disrupted by fixed-window segmentation.

### 2.3 Why all-MiniLM-L6-v2

The embedding model needed to satisfy two constraints: sufficient retrieval quality to make the evaluation meaningful, and computational cost compatible with local evaluation on an M1 Mac with 8GB RAM.

all-MiniLM-L6-v2 is a distilled sentence transformer that produces 384-dimensional embeddings. It is validated on the BEIR retrieval benchmark and represents a well-understood performance baseline in the RAG literature (Gao et al., 2023). Its 384-dimensional output is sufficient for the semantic similarity task without the overhead of larger models such as text-embedding-3-large. The full QASPER corpus can be indexed in under 100MB RAM, making local development feasible.

Critically, the embedding model is held constant across all three variants. Any performance difference between V0, V1, and V2 is therefore not attributable to the embedding model.

### 2.4 Why FAISS IndexFlatL2

FAISS IndexFlatL2 performs exact nearest-neighbour search using L2 distance. For L2-normalised vectors, L2 distance and cosine similarity produce identical rankings. Exact search was chosen over approximate methods (HNSW, IVF) to eliminate approximation error as a confounding variable. If the retrieval mechanism were approximate, observed differences between variants could partly reflect approximation variance rather than the experimental intervention.

---

## 3. Graph Construction

### 3.1 Node Types

**Paragraph nodes** - one per paragraph chunk from V1. Each node stores the paper ID, section name, paragraph index, and chunk text. The paragraph index within a section is the primary ordering signal for sequential edge construction.

**Virtual section nodes** - one per (paper_id, section_name) pair. These are not real text chunks - they are structural aggregation nodes that allow the graph to represent section-level relationships without requiring paragraph-to-paragraph edges across sections. A paragraph that is the only member of its section connects directly to its section node with no sequential edges.

### 3.2 Edge Types

**Sequential edges** connect paragraph N to paragraph N+1 within the same section of the same paper. These edges encode narrative continuity - the assumption that adjacent paragraphs in the same section are more likely to contain related evidence than non-adjacent paragraphs. This assumption holds strongly for scientific papers where arguments build sequentially.

**Section membership edges** connect each paragraph node to its virtual section node. These edges enable section-level traversal - if the anchor node is in the methodology section, its section node connects to all other methodology paragraphs, making them reachable within 2 hops even if they are not immediately adjacent.

**No inter-paper edges.** QASPER questions target single papers. Cross-paper edges would introduce noise without adding retrieval value for within-paper evidence retrieval.

### 3.3 Graph Statistics

The final graph over the full QASPER train split contains approximately 46,882 paragraph nodes, 12,418 virtual section nodes, and 81,346 edges. The graph is constructed once and serialised as a NetworkX pickle file (26MB) for reuse across all V2 queries. Reconstruction takes approximately 8 minutes on an M1 Mac.

---

## 4. V2 Retrieval Mechanism

### 4.1 Why Widen Initial Retrieval to K=10

V0 and V1 retrieve top-5 directly. V2 widens the initial retrieval to top-10 before applying the graph filter. This is necessary because the graph filter re-scores and potentially suppresses some of the initial candidates. If the initial retrieval were top-5, the graph filter might reduce the final set below 5 after applying penalties. Starting with 10 guarantees the graph filter has sufficient candidates to produce a full top-5 final set even after suppression.

### 4.2 Anchor Selection

The top-1 candidate by cosine similarity becomes the anchor node. The assumption is that the highest-similarity chunk is the strongest signal for where in the document the answer is located. Graph re-scoring then uses this anchor to identify structurally adjacent content that cosine similarity may have ranked lower or missed entirely.

An alternative approach would be to use all top-K candidates as anchors and union their 2-hop neighbourhoods. This was considered but rejected because it would produce a proximity zone that expands with K, making the graph constraint parameter-dependent in a complex way. Single-anchor traversal keeps the mechanism transparent and interpretable.

### 4.3 Graph Re-scoring

For each candidate in the initial top-10, the shortest path length from the anchor is computed using NetworkX's `shortest_path_length`. This is O(V+E) per query via BFS. For the graph sizes involved (~59K nodes, ~81K edges), this takes under 50ms per query.

Candidates within HOP_LIMIT hops receive a score bonus. Candidates outside HOP_LIMIT or with no path to the anchor receive a penalty multiplier. The final score is:

```
Within HOP_LIMIT:  final_score = cosine_sim + PROXIMITY_BONUS
Outside HOP_LIMIT: final_score = cosine_sim × PENALTY
No path:           final_score = cosine_sim × PENALTY
```

### 4.4 Adjacency Expansion

After re-scoring, sequential graph neighbours of the anchor that were not in the initial top-10 FAISS results are added to the candidate set. These are chunks that cosine similarity did not rank in the top-10 but that are immediately adjacent to the highest-confidence retrieval result in the document. The assumption is that a chunk immediately following the anchor paragraph is more likely to contain continuation evidence than a random chunk ranked 8th by embedding similarity.

Adjacency expansion candidates receive a final score of `cosine_sim + PROXIMITY_BONUS` where cosine_sim is computed on demand by embedding the candidate chunk and computing L2 distance to the query vector.

---

## 5. Parameter Selection - Ablation Study

All three V2 graph constraint parameters were selected via a 30-query ablation study. Each parameter was varied independently while the others were held at candidate values.

### 5.1 HOP_LIMIT

**Values tested:** 1, 2, 3  
**Selected value:** 2  
**Rationale:**

HOP_LIMIT=1 recovers only immediately adjacent paragraphs - the paragraph immediately before and after the anchor. For a question whose answer spans two paragraphs, this is sufficient. For a question whose answer requires a methodology detail from three paragraphs before the results paragraph that triggered the anchor, it is not. In practice, scientific paper arguments frequently span three to four paragraphs, making a single-hop neighbourhood too narrow.

HOP_LIMIT=3 begins pulling in paragraphs from adjacent sections via the virtual section node path. A 3-hop traversal from a results paragraph can reach introduction paragraphs via the section node. These are structurally distant and typically irrelevant to the specific question - they introduce noise rather than context.

HOP_LIMIT=2 covers the immediate neighbourhood plus one level of extension. For a paper with sections of 4-6 paragraphs, this typically covers the full local argument context without reaching into unrelated sections. The selection is additionally supported by Wang et al. (2023), who use 2-hop traversal as the standard configuration in Knowledge Graph Prompting.

### 5.2 PROXIMITY_BONUS

**Values tested:** 0.1, 0.3, 0.5  
**Selected value:** 0.3  
**Rationale:**

PROXIMITY_BONUS=0.1 is insufficient to overcome the cosine similarity score advantage of high-ranking distant chunks. If a chunk with cosine_sim=0.72 is outside the proximity zone and a structurally adjacent chunk has cosine_sim=0.65, a bonus of 0.1 produces final scores of 0.72 and 0.75 - a marginal advantage that gets wiped out by the penalty applied to the distant chunk. The net effect at 0.1 is too small to meaningfully alter rankings.

PROXIMITY_BONUS=0.5 over-weights structural proximity relative to semantic similarity. A chunk with cosine_sim=0.40 that happens to be adjacent to the anchor receives a final score of 0.90, outranking chunks with cosine_sim=0.70 that contain the actual answer. At 0.5, the graph constraint begins overriding the embedding signal rather than complementing it.

PROXIMITY_BONUS=0.3 produces a meaningful but not dominant boost. A structurally adjacent chunk needs approximately cosine_sim ≥ 0.40 to rank in the final top-5, which filters out clearly irrelevant adjacent chunks while surfacing contextually relevant ones.

### 5.3 PENALTY

**Values tested:** ×0.25, ×0.5, ×0.75  
**Selected value:** ×0.5  
**Rationale:**

PENALTY=×0.25 is too aggressive. A chunk with cosine_sim=0.70 that falls outside the proximity zone receives a final score of 0.175 - effectively removed from contention entirely. This causes the graph constraint to behave more like a hard filter than a soft re-ranker, which is undesirable when a high-similarity chunk outside the proximity zone may genuinely contain the answer (for example, if the question targets a section far from the anchor's section).

PENALTY=×0.75 is too lenient. Noisy chunks with cosine_sim=0.65 outside the proximity zone receive 0.49, which still competes effectively with proximity-boosted chunks. The penalty is not strong enough to meaningfully suppress structural noise.

PENALTY=×0.5 halves the score of out-of-zone chunks. A chunk with cosine_sim=0.70 outside the zone receives 0.35, which ranks below any proximity-zone chunk with cosine_sim ≥ 0.05. This effectively prioritises structural relevance while retaining high-similarity out-of-zone chunks as fallbacks when the proximity zone is empty or low-quality.

---

## 6. Evaluation Framework

### 6.1 Why RAGAS

RAGAS (Es et al., 2024) was selected because it provides component-level metrics that separate retrieval failures from generation failures. A single end-to-end accuracy score conflates these two failure surfaces - a system that retrieves correctly but generates poorly looks identical to one that generates correctly from bad evidence. Since this study is specifically testing a retrieval-layer intervention, separating retrieval quality from generation quality is not optional.

RAGAS provides four metrics aligned to the two failure surfaces:
- Context Recall and Context Precision target retrieval quality
- Faithfulness and Answer Relevancy target generation quality

The reference-free design is important practically. QASPER provides ground truth answers but annotating every retrieval decision at scale by hand is not feasible. RAGAS uses an LLM judge to score each metric automatically.

### 6.2 Why Provenance Coverage

Provenance Coverage was added as a supplementary V2-only metric because the RAGAS metrics measure outcomes - whether the right evidence was retrieved - but not mechanism. Provenance Coverage measures whether the graph constraint is actually doing what it was designed to do: surfacing structurally connected content.

A Provenance Coverage of 0.48 means that 48% of the final retrieved chunks have a traceable structural path to the anchor node within the 2-hop neighbourhood. This directly measures the graph constraint's operational contribution, independent of whether the retrieved content helped the LLM answer correctly.

### 6.3 RAGAS API Compatibility Issue

During the full 50-query evaluation run, RAGAS LLM-dependent metrics returned 0.0 across all queries. The root cause was identified as an API version incompatibility - `EvaluationResult` no longer supports the `.get()` method used in the `safe_score()` wrapper. The correct extraction method for the updated RAGAS version is `.to_pandas()`.

The fix:

```python
# Broken - .get() no longer supported on EvaluationResult
val = result.get("context_recall")

# Fixed - convert to DataFrame first
result_df = result.to_pandas()
val = result_df["context_recall"].iloc[0]
```

This issue was identified after the evaluation run completed. Primary metric evidence is therefore drawn from the 10-query pilot evaluation, which was conducted under a verified configuration using Llama 3.2 3B via Ollama. Provenance Coverage, computed from graph metadata without LLM involvement, was successfully evaluated across the full 50-query run.

---

## 7. Known Limitations and Mitigations

### 7.1 Pilot Sample Size (n=10)

The pilot evaluation uses 10 queries, which is insufficient for statistical reliability. The Wilcoxon signed-rank test planned for the full evaluation cannot be applied meaningfully at n=10. The pilot results should be treated as directional evidence - they establish which variant performs better on each metric but the magnitude of the differences should not be over-interpreted.

**Mitigation:** The stratified sampling strategy ensures the 10 queries cover extractive, abstractive, and yes/no question types in the right proportions. The directional finding - V2 outperforms V0 on Context Recall - is consistent with the architectural logic of the design and with Wang et al. (2023), which increases confidence that the direction is real even if the magnitude is uncertain.

### 7.2 Single Dataset

The evaluation is conducted on QASPER only. QASPER's NLP research paper domain and information-seeking question type are well-matched to the study's structural failure modes, but the results cannot be generalised to other domains without further evaluation.

**Mitigation:** QASPER was selected for principled reasons tied to the specific failure modes under investigation, not for convenience. The generalisability limitation is acknowledged explicitly in the report and identified as future work.

### 7.3 Groq API Rate Limiting

The full evaluation was run via the Groq API free tier (30 requests/minute for Llama 3.1 8B). A 12-second sleep between API calls was required to stay under the rate limit. This extended the evaluation runtime significantly and introduced network latency variability into generation timing.

**Mitigation:** Incremental saving after every query ensured no results were lost to interruption. The sleep timer was set conservatively enough that no queries failed due to rate limiting after the timer was adjusted from 2 seconds to 12 seconds at query 100 of the V0 run.

---

## 8. Reproducibility Notes

- All random operations use `seed=42` throughout - query sampling, stratified split, and any shuffle operations
- The embedding model is loaded with `normalize_embeddings=True` to ensure L2 distance and cosine similarity are equivalent
- Temperature is set to 0 on the generation model to ensure deterministic outputs across runs
- The FAISS index uses `IndexFlatL2` (exact search) to eliminate approximation variance
- The NetworkX graph is serialised as a pickle file after construction and loaded for all V2 queries - graph is never rebuilt mid-experiment
- Pre-built artefacts (FAISS indexes, chunk files, graph) are stored in `notebooks/data/` and load automatically in `full_evaluation.ipynb`
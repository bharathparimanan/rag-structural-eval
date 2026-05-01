# Known Issues Log

## Graph-Constrained Vector Retrieval - MSc Dissertation 7PAM2002

A chronological record of technical issues encountered throughout the project, how each was diagnosed, and how it was resolved or mitigated. Maintained as a transparency record for reproducibility.

---

## Issue 1 - QASPER Dataset Loading Failure

**When:** Early development - dataset loading stage  
**Severity:** Blocking  
**Status:** Resolved

### What Happened
Loading QASPER via `load_dataset("allenai/qasper")` failed with a Parquet format error. The default dataset configuration was not compatible with the version of the datasets library being used.

### Root Cause
QASPER requires a specific revision string to load the correctly formatted Parquet version from Hugging Face.

### Fix
```python
ds = load_dataset(
    "allenai/qasper",
    revision="refs/convert/parquet",
    trust_remote_code=False
)
```

### Impact
No impact on results. Required one-line fix applied to all notebooks.

---

## Issue 2 - DATA_DIR Path Resolution Failure

**When:** Full evaluation notebook development  
**Severity:** Blocking  
**Status:** Resolved

### What Happened
The full evaluation notebook could not find artefact files (`queries.json`, `v0_chunks.json` etc.) because the `DATA_DIR` path resolved differently depending on whether Jupyter was launched from the project root or from inside the `notebooks/` directory.

### Root Cause
Hardcoded `DATA_DIR = "data"` resolved to different absolute paths depending on the working directory at Jupyter launch time.

### Fix
```python
import pathlib
_cwd = pathlib.Path.cwd()
if (_cwd / "notebooks" / "data").exists():
    DATA_DIR = str(_cwd / "notebooks" / "data")
elif (_cwd / "data").exists():
    DATA_DIR = str(_cwd / "data")
else:
    DATA_DIR = str(_cwd / "data")
```

### Impact
No impact on results. Path detection added to Cell 8 of `full_evaluation_groq.ipynb`.

---

## Issue 3 - queries.json Had Only 10 Queries

**When:** Full evaluation run setup  
**Severity:** Blocking  
**Status:** Resolved

### What Happened
When the full evaluation notebook tried to load `queries.json`, it found only 10 queries - the pilot sample built during V0 notebook development. The full evaluation requires 50.

### Root Cause
The pilot notebooks built `queries.json` with `N_PILOT = 10`. This file was never regenerated for the full evaluation.

### Fix
A query regeneration cell was added to `full_evaluation_groq.ipynb` that reloads QASPER and generates 50 stratified queries using stratified sampling (50% extractive, 30% abstractive, 20% yes/no, seed=42). This overwrites the 10-query pilot file.

### Impact
No impact on results. Query regeneration cell runs before artefact loading.

---

## Issue 4 - Groq API Rate Limit Failures at Query 100

**When:** V0 full evaluation run (first attempt)  
**Severity:** High - caused query failures  
**Status:** Resolved

### What Happened
During the V0 evaluation run, consistent rate limit errors (HTTP 429) began occurring at approximately query 100. The initial `SLEEP_BETWEEN_CALLS = 2` seconds was insufficient.

### Root Cause
Each query makes approximately 5 Groq API calls total - 1 for generation and 4 for RAGAS metric scoring. At 2 seconds per call, this equals ~10 calls per minute, which was fine early in the run but triggered rate limiting as the session accumulated requests.

### Fix
`SLEEP_BETWEEN_CALLS` increased from 2 seconds to 12 seconds in Cell 8 of `full_evaluation_groq.ipynb`. This distributes calls to approximately 5 per minute, safely under the 30 request/minute free tier limit.

```python
SLEEP_BETWEEN_CALLS = 12  # seconds between each Groq API call
```

### Impact
Queries that failed due to rate limiting were marked as `status: failed` in the results file. The resume logic re-ran these on the next session. No data was permanently lost due to the incremental save mechanism.

---

## Issue 5 - Hardcoded API Key Committed to GitHub

**When:** GitHub push during notebook development  
**Severity:** Critical - security risk  
**Status:** Resolved

### What Happened
A Groq API key was hardcoded directly in a notebook cell and committed to the repository. GitHub's push protection detected the secret and blocked the push.

### Root Cause
During debugging, the API key was temporarily hardcoded to test the Groq connection. The notebook was committed before the key was removed.

### Resolution Steps
1. Rotated the exposed key immediately at console.groq.com
2. Attempted `git rebase -i` to remove the commit - encountered rebase-merge directory conflict
3. Used GitHub's secret scanning unblock URL to allow the push after confirming the key was already rotated
4. Replaced hardcoded key with `python-dotenv` pattern in all subsequent notebook versions

### Prevention
`.env` file added to `.gitignore`. `.env.example` template committed showing required structure. All notebooks now load the API key via:
```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
```

### Impact
No security risk after key rotation. The exposed key was deleted before any unauthorised use could occur.

---

## Issue 6 - RAGAS LLM-Dependent Metrics Returning 0.0

**When:** Full evaluation run - all three variants  
**Severity:** Critical - invalidated all LLM-dependent metric results  
**Status:** Identified, fix applied, re-run required

### What Happened
Context Recall, Context Precision, Faithfulness, and Answer Relevancy all returned 0.0 across every query for all three variants. Provenance Coverage, which does not use the LLM judge, returned correct values (V2: 0.4776).

### Root Cause
A RAGAS API version incompatibility. The `EvaluationResult` object returned by the updated RAGAS framework no longer supports `.get()` for score extraction. The original `safe_score()` function was calling `result.get("context_recall")` which raised an `AttributeError` silently caught by the except block, returning 0.0 for every metric.

```python
# Broken - raises AttributeError in updated RAGAS
val = result_dict.get(key)

# Fixed - correct API for updated RAGAS
df = result.to_pandas()
val = df[col].iloc[0]
```

### Fix Applied
`evaluate_with_ragas()` in Cell 15 of `full_evaluation_groq.ipynb` was rewritten to use `.to_pandas()` for score extraction. `safe_score()` was moved inline and now operates on a DataFrame column rather than a dict key.

### Additional Complication
When copying the fixed code from the chat interface into Jupyter, the line `df = result.to_pandas()` was rendered as a hyperlink `df = [result.to](http://result.to)_pandas()` by the markdown renderer. This required a programmatic fix using `nbformat` to replace the corrupted line directly in the notebook file.

```python
import nbformat
nb = nbformat.read(notebook_path, as_version=4)
for cell in nb.cells:
    if '[result.to]' in cell.source:
        cell.source = cell.source.replace(
            'df = [result.to](http://result.to)_pandas()',
            'df = result.to_pandas()'
        )
nbformat.write(nb, notebook_path)
```

### Impact
All three result files (`full_eval_v0_results.json`, `full_eval_v1_results.json`, `full_eval_v2_results.json`) from the original run contain zeroed LLM-dependent metrics and must be deleted before re-running. Provenance Coverage results are valid and retained. The pilot evaluation results (n=10) are used as primary metric evidence in the report pending the corrected full evaluation run.

### Mitigation in Report
Documented honestly in Section 5.1 and Section 6.4 of the Final Project Report. Pilot results (n=10) reported as primary metric evidence. Provenance Coverage from the full 50-query run reported as supplementary structural evidence.

---

## Issue 7 - Llama 3.2 3B JSON Parse Failures in Pilot

**When:** Pilot evaluation (10-query run using Ollama)  
**Severity:** Medium - affected V1 and V2 metric reliability  
**Status:** Resolved by model upgrade

### What Happened
During the pilot evaluation using Llama 3.2 3B via Ollama, Context Precision and Faithfulness scores for V1 and V2 were unreliable. The model consistently failed to return valid JSON responses required by RAGAS for these metrics.

### Root Cause
Llama 3.2 3B is too small to reliably follow RAGAS's structured JSON output instructions. The model would return natural language responses instead of the required JSON format, causing RAGAS to return NaN scores handled by `safe_score()` as 0.0.

### Fix
Switched to Llama 3.1 8B via Groq API for the full evaluation. The larger model follows structured output instructions reliably. V0 pilot scores for Context Precision (0.270) and Faithfulness (0.083) are reported with caution as they came from a small number of queries where valid JSON was returned before the failure pattern became consistent.

### Impact
V1 and V2 Context Precision and Faithfulness scores from the pilot are marked as unreliable (-) in Table 4 of the report. This is acknowledged as a limitation in Section 6.4.

---

## Issue 8 - Rebase-Merge Directory Conflict

**When:** Attempting to remove API key from Git history  
**Severity:** Low - process blocker  
**Status:** Resolved

### What Happened
When attempting `git rebase -i` to remove the commit containing the hardcoded API key, the rebase failed with:
```
fatal: It seems that there is already a rebase-merge directory
```
A previous incomplete rebase had left the `.git/rebase-merge` directory in place.

### Fix
```bash
git rebase --abort
rm -fr ".git/rebase-merge"
git stash
git rebase -i <commit>~1
```

### Impact
None. Resolved before the push was completed.

---

## Summary Table

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | QASPER dataset loading failure | Blocking | ✓ Resolved |
| 2 | DATA_DIR path resolution failure | Blocking | ✓ Resolved |
| 3 | queries.json had only 10 queries | Blocking | ✓ Resolved |
| 4 | Groq rate limit failures at query 100 | High | ✓ Resolved |
| 5 | Hardcoded API key committed to GitHub | Critical | ✓ Resolved |
| 6 | RAGAS metrics returning 0.0 (.get() bug) | Critical | ⚠ Fix applied, re-run in progress |
| 7 | Llama 3.2 3B JSON parse failures in pilot | Medium | ✓ Resolved (model upgrade) |
| 8 | Rebase-merge directory conflict | Low | ✓ Resolved |

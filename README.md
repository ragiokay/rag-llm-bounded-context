# RAG-LLM Bounded Context Extractor

Automated Bounded Context creation through a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG).

---

## Pipeline Overview

```
Qdrant (vector DB, pre-seeded on server)
↓ retrieve.py: query_similar / query_all
Prompt Generator
↓ augmented prompts
vLLM (Llama-3.2-1B-Instruct)
↓
Bounded Context output (JSON)
```

---

## For Teammates: How to Use the Retriever

The retriever is already seeded and ready on the shared Qdrant server.
Import `retrieve.py` directly — no local setup needed.

### Environment Variables (set by Salvador in the shared container)

| Variable | Value |
|---|---|
| `QDRANT_URL` | `http://140.112.90.146:6333` |
| `COLLECTION_PREFIX` | `spring2026SE_g1_rag_` |

### Import

```python
import sys
sys.path.insert(0, "embedding")   # adjust path as needed
from retrieve import query_similar, query_all, query_multiple, query_by_prefix
```

---

### `query_similar(query_text, collection, n_results)`

Search a single collection.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | `str` | required | Natural language query |
| `collection` | `str` | `"maven_ere_causal"` | Collection name (with prefix) |
| `n_results` | `int` | `3` | Number of results to return |

```python
results = query_similar(
    "The storm caused severe flooding.",
    collection="spring2026SE_g1_rag_maven_ere_causal",
    n_results=3,
)
```

---

### `query_all(query_text, n_results_per_collection)`

Search all collections at once, results sorted by similarity.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | `str` | required | Natural language query |
| `n_results_per_collection` | `int` | `3` | Results per collection |

```python
results = query_all("The drought caused crop failure.", n_results_per_collection=2)
```

---

### `query_multiple(query_text, collections, n_results_per_collection)`

Search a specific list of collections and return merged results sorted by distance.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | `str` | required | Natural language query |
| `collections` | `list[str]` | required | Exact collection names to search |
| `n_results_per_collection` | `int` | `3` | Results per collection |

```python
results = query_multiple(
    "User cancels a scheduled meeting.",
    collections=["spring2026SE_g1_rag_mtop_commands",
                 "spring2026SE_g1_rag_log_commands"],
    n_results_per_collection=3,
)
```

---

### `query_by_prefix(query_text, prefixes, n_results_per_collection)`

Search all collections whose names match any of the given prefixes.
`COLLECTION_PREFIX` is automatically prepended to each prefix.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_text` | `str` | required | Natural language query |
| `prefixes` | `list[str]` | required | Collection name prefixes to match (without `COLLECTION_PREFIX`) |
| `n_results_per_collection` | `int` | `3` | Results per collection |

```python
# Search all log_* collections (log_domain_events + log_commands)
results = query_by_prefix("The meeting is cancelled.", prefixes=["log"])

# Search both bpc_* and log_* collections
results = query_by_prefix("User submits a form.", prefixes=["bpc", "log"])
```

---

### Return Format

Both functions return a `list[dict]`. Each item:

```json
{
  "input": "The sentence that was embedded (matched document)",
  "distance": 0.43,
  "collection": "spring2026SE_g1_rag_maven_ere_causal",
  "output": {
    "policy": "CAUSE: famine → deaths",
    "domain_event": "Many civilian deaths were caused by famine due to war conditions.",
    "command": "Many civilian deaths were caused by famine due to war conditions.",
    "bounded_context": "World War II",
    "aggregate": "Die",
    "source_phrase": "Many civilian deaths were caused by famine due to war conditions.",
    "trigger_span": null,
    "views": null,
    "user_roles": null,
    "process": null
  }
}
```

| Field | Description |
|---|---|
| `input` | The matched document text |
| `distance` | Cosine distance (0 = identical, 1 = unrelated) — lower is better |
| `collection` | Which collection the result came from |
| `output.policy` | Causal/command relation label |
| `output.domain_event` | Cause sentence or event description |
| `output.command` | Effect sentence or command name |
| `output.bounded_context` | Domain / source context |
| `output.aggregate` | Aggregate or event type |
| `output.source_phrase` | Original source text |
| `output.trigger_span` | Imperative trigger phrase (MTOP only) |

---

## Available Collections (pre-seeded)

All collections use prefix `spring2026SE_g1_rag_`.

| Collection | Dataset | Vectors | Description |
|---|---|---|---|
| `bpc_education` | BPC | 327 | Business process causal reasoning — education |
| `bpc_finance` | BPC | 391 | Finance |
| `bpc_human_resources` | BPC | 381 | Human resources |
| `bpc_insurance` | BPC | 272 | Insurance |
| `bpc_logistics` | BPC | 296 | Logistics |
| `bpc_manufacturing` | BPC | 364 | Manufacturing |
| `bpc_medical` | BPC | 345 | Medical |
| `bpc_retail` | BPC | 232 | Retail |
| `bpc_transportation` | BPC | 469 | Transportation |
| `maven_ere_causal` | MAVEN-ERE | 36,316 | Event cause/precondition pairs |
| `mtop_commands` | MTOP | 11,159 | Task-oriented command utterances |
| `log_domain_events` | DDD Log | 85 | Domain events extracted from DDD extraction tool log |
| `log_commands` | DDD Log | 69 | Commands with actors extracted from DDD extraction tool log |

---

## Project Structure

```
rag-llm-bounded-context/
├── embedding/
│   ├── retrieve.py          # Retrieval interface (query_similar / query_all / query_multiple / query_by_prefix)
│   ├── embed.py             # BPC seeding script
│   ├── seed_maven_ere.py    # MAVEN-ERE seeding script
│   ├── seed_mtop.py         # MTOP seeding script
│   ├── seed_log.py          # DDD Log seeding script (domain events + commands)
│   ├── reset_collections.py # Maintenance: list / delete collections by prefix
│   ├── causal_transform.py  # BPC row → CausalRelationRecord (DDD schema)
│   └── requirements.txt
├── tests/                   # 202 unit tests (all use in-memory Qdrant)
│   ├── test_log.py          # Parser tests for seed_log.py
│   └── test_retrieve_log.py # Retrieval tests for log collections
├── reports/                 # Auto-generated demo output (retrieve.py __main__)
│   ├── retrieve_demo_*.md   # Human-readable results table
│   └── retrieve_demo_*.json # Raw JSON results by query
├── Dockerfile
└── README.md
```

---

## Re-seeding (if needed)

The datasets are already seeded. Only re-run if the server data is lost.

```bash
# Set environment variables first
export QDRANT_URL=http://140.112.90.146:6333
export COLLECTION_PREFIX=spring2026SE_g1_rag_

# BPC (downloads from HuggingFace automatically)
python embedding/embed.py

# MAVEN-ERE (needs data/maven_ere/train.jsonl, or falls back to HuggingFace)
python embedding/seed_maven_ere.py

# MTOP (needs data/mtop/en/train.txt — download from https://fb.me/mtop_dataset)
python embedding/seed_mtop.py

# DDD Log (needs data/log/*.log — produced by the automated DDD extraction tool)
python embedding/seed_log.py
```

To inspect or clean up collections:

```bash
python embedding/reset_collections.py --list
python embedding/reset_collections.py --prefix spring2026SE_g1_rag_
```

---

## Running Tests

```bash
pip install -r embedding/requirements.txt
python -m pytest tests/ -q
```

All tests use in-memory Qdrant — no server connection required.

To generate a full demo report across all collections (requires Qdrant running):

```bash
python embedding/retrieve.py
# Writes reports/retrieve_demo_<timestamp>.md  (readable table)
# Writes reports/retrieve_demo_<timestamp>.json (raw results)
```

---

## Requirements Traceability

| Code | Requirement | Implemented In |
|---|---|---|
| BFR5 | Organize and clean datasets | `embedding/embed.py`, `causal_transform.py` |
| BFR6 | Transfer datasets into vector space | `embedding/embed.py`, `seed_maven_ere.py`, `seed_mtop.py`, `seed_log.py` |
| BFR7 | Review and clean vector space | `embedding/embed.py` (`review_collection`) |
| IIR3 | Embedding Module ↔ Vector Database | `embedding/retrieve.py` |

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

## Developer Guide: How to Add a New DDD Module (e.g., Commands)

### Step 1: Create the Filter (`filters/CommandsFilter.py`)
```python
class CommandsFilter:
    def __init__(self, retriever):
        self.retriever = retriever
        self.target_collection = "mtop_commands" # Adjust based on Qdrant collections

    def get_clean_examples(self, query_text: str, top_k: int = 3) -> list:
        raw_results = self.retriever.search(query_text, self.target_collection, top_k)
        return [item["output"] for item in raw_results if "output" in item]
```

### Step 2: Create the Prompt Generator (prompt_generator/CommandsPromptGenerator.py)
```python
from prompt_generator.BasePromptGenerator import BasePromptGenerator

class CommandsPromptGenerator(BasePromptGenerator):
    def get_role(self) -> str:
        return "You are a strict JSON API server. You ONLY output raw JSON."
    def get_task(self) -> str:
        return "Identify the Commands from the text."
    def get_rules(self) -> str:
        return "CRITICAL: MUST output valid JSON. DO NOT output multiple JSON blocks."
    def get_output_schema(self) -> dict:
        return {"Commands": ["PlaceOrder", "ProcessPayment"]}
```

### Step 3: Create the Service Class (rag_llm_module/NL2IdentifyCommands.py)
```python
from rag_llm_module.BaseNL2Service import BaseNL2Service
from prompt_generator.CommandsPromptGenerator import CommandsPromptGenerator

class NL2IdentifyCommands(BaseNL2Service):
    def get_generator(self):
        return CommandsPromptGenerator()
```
### Step 4: Notice the valid keys (rag_llm_module/ResponseParser.py)
```python
valid_keys = ["BusinessLogic", "DomainEvents", "Commands", "Actors"]
```

### Step 5: Adjust main for output format

### Import

```python
import sys
sys.path.insert(0, "embedding")   # adjust path as needed
from retrieve import query_similar, query_all
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
| `maven_ere_causal` | MAVEN-ERE | 11,000+ | Event cause/precondition pairs |
| `mtop_commands` | MTOP | 11,159 | Task-oriented command utterances |

---

## Project Structure

```
rag-llm-bounded-context/
├── embedding/
│   ├── retrieve.py          # Retrieval interface (query_similar, query_all)
│   ├── embed.py             # BPC seeding script
│   ├── seed_maven_ere.py    # MAVEN-ERE seeding script
│   ├── seed_mtop.py         # MTOP seeding script
│   ├── reset_collections.py # Maintenance: list / delete collections by prefix
│   ├── causal_transform.py  # BPC row → CausalRelationRecord (DDD schema)
│   └── requirements.txt
├── tests/                   # 162 unit tests (all use in-memory Qdrant)
├── Dockerfile
├── main.py 
├── filters/
│   ├── PromptBuilder.py
│   ├── BasePromptGenerator.py
│   └── DomainEventsPromptGenerator.py
├── prompt_generator/                 
│   ├── PromptBuilder.py
│   ├── BasePromptGenerator.py
│   ├── BusinessLogicPromptGenerator.py
│   └── DomainEventsPromptGenerator.py
│
└── rag_llm_module/                   
│   ├── LLMConnector.py       
│   ├── BaseNL2Service.py
│   ├── NL2IdentifyBusinessLogic.py
│   └── NL2IdentifyDomainEvents.py
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

---

## Requirements Traceability

| Code | Requirement | Implemented In |
|---|---|---|
| BFR5 | Organize and clean datasets | `embedding/embed.py`, `causal_transform.py` |
| BFR6 | Transfer datasets into vector space | `embedding/embed.py`, `seed_maven_ere.py`, `seed_mtop.py` |
| BFR7 | Review and clean vector space | `embedding/embed.py` (`review_collection`) |
| LFR1 | Prompt Element Assembly | `prompt_generator/PromptBuilder.py, filters/` |
| LFR2 | LLM Connection & JSON Parsing | `rag_llm_module/LLMConnector.py (ResponseParser)` |
| LFR3 | Domain Element Identification | `rag_llm_module/NL2Identify*.py` |
| LFR4 | Architectural Aggregation | `rag_llm_module/ (Future Aggregation classes)` |
| IIR2 | Prompt Token validation | Implicitly handled via top_k limiting in filters/ |
| IIR3 | Embedding Module ↔ Vector Database | `embedding/retrieve.py` |
| IIR4 | Prompt Generator ↔ RAG-LLM Interface | `rag_llm_module/BaseNL2Service.py (get_generator)` |

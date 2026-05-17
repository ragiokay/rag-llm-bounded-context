
# RAG-LLM Bounded Context Extractor

Automated Bounded Context creation through a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG).

---

## What This Project Does

This system takes a software use case description (in JSON format) and automatically identifies **Bounded Contexts** from it, using an LLM augmented with NLP reference datasets.

### Pipeline Overview
```
ChromaDB (vector DB)
↓ Prompt Generator queries similar context
Prompt Generator
↓ Sends augmented prompts
RAG-LLM
↓ Identifies Bounded Contexts
Bounded Context Module
↓ Returns structured JSON output
User
```
---

## Project Structure

```
rag-llm-bounded-context/
├── embedding/              # Embedding Module
│   ├── embed.py            # Downloads BPC from HuggingFace, embeds into ChromaDB
│   ├── seed_maven_ere.py   # Loads MAVEN-ERE, embeds into ChromaDB
│   ├── seed_mtop.py        # Loads MTOP, embeds into ChromaDB
│   ├── retrieve.py         # Retrieval interface for Prompt Generator
│   ├── causal_transform.py # BPC row → CausalRelationRecord (DDD schema)
│   ├── reset_collections.py
│   └── requirements.txt
├── tests/
├── .gitignore
└── README.md
```
---

## Requirements

- Python 3.11+
- Git

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/ragiokay/rag-llm-bounded-context.git
cd rag-llm-bounded-context
```

### 2. Install dependencies

```bash
pip install -r embedding/requirements.txt
```

### 3. Build the vector database

```bash
python embedding/embed.py
```

This downloads the BPC dataset from HuggingFace (3,077 rows), converts all records into vectors using `all-MiniLM-L6-v2`, and stores them in `vector_db/`. 9 collections are created, one per domain.

Optionally, add the MAVEN-ERE and MTOP datasets:

```bash
# MAVEN-ERE (requires local data/maven_ere/train.jsonl, or falls back to HuggingFace mirror)
python embedding/seed_maven_ere.py

# MTOP (requires local data/mtop/en/train.txt — download from https://fb.me/mtop_dataset)
python embedding/seed_mtop.py
```

### 4. Test retrieval

```bash
python embedding/retrieve.py
```

---

## Datasets

| Dataset   | Source                                                      | Rows  | Domain                           |
|-----------|-------------------------------------------------------------|-------|----------------------------------|
| BPC       | [ibm-research/BPC](https://huggingface.co/datasets/ibm-research/BPC) | 3,077 | Business Process Causal Reasoning |
| MAVEN-ERE | [THU-KEG/MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE)  | —     | Event Causal Relations           |
| MTOP      | [fb.me/mtop_dataset](https://fb.me/mtop_dataset)           | —     | Task-Oriented Parsing (commands) |

---

## ChromaDB Collections

After running `embed.py`, the following collections are available in `vector_db/`:

| Collection             | Domain           | Vectors |
|------------------------|------------------|---------|
| bpc_education          | Education        | 327     |
| bpc_finance            | Finance          | 391     |
| bpc_human_resources    | Human Resources  | 381     |
| bpc_insurance          | Insurance        | 272     |
| bpc_logistics          | Logistics        | 296     |
| bpc_manufacturing      | Manufacturing    | 364     |
| bpc_medical            | Medical          | 345     |
| bpc_retail             | Retail           | 232     |
| bpc_transportation     | Transportation   | 469     |

After running `seed_maven_ere.py`:

| Collection        | Domain              |
|-------------------|---------------------|
| maven_ere_causal  | Event causal pairs  |

After running `seed_mtop.py`:

| Collection     | Domain                  |
|----------------|-------------------------|
| mtop_commands  | Task-oriented commands  |

---

## Notes

- `vector_db/` is not committed to Git. Generate it locally by running the seed scripts.
- `data/` (raw dataset files) is not committed to Git.

---

## Requirements Traceability

| Code | Requirement                       | Implemented In              |
|------|-----------------------------------|-----------------------------|
| BFR5 | Organize and clean datasets       | `embedding/embed.py`        |
| BFR6 | Transfer datasets into vector space | `embedding/embed.py`      |
| BFR7 | Review and clean vector space     | `embedding/embed.py`        |
| IIR3 | Embedding Module ↔ Vector Database | `embedding/embed.py`       |

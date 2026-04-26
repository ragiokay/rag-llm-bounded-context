
# RAG-LLM Bounded Context Extractor

Automated Bounded Context creation through a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG).

---

## What This Project Does

This system takes a software use case description (in JSON format) and automatically identifies **Bounded Contexts** from it, using an LLM augmented with NLP reference datasets.

### Pipeline Overview

MySQL (relational DB) 
↓ Embedding Module reads datasets 
ChromaDB (vector DB) 
↓ Prompt Generator queries similar context 
Prompt Generator 
↓ Sends augmented prompts 
RAG-LLM 
↓ Identifies Bounded Contexts 
Bounded Context Module 
↓ Returns structured JSON output 
User

---

## Project Structure

```
rag-llm-bounded-context/
├── database/               # Database Module (BFR1–BFR4)
│   ├── schema.sql          # MySQL schema definition
│   ├── seed.py             # Downloads BPC dataset and loads into MySQL
│   └── requirements.txt
├── embedding/              # Embedding Module (BFR5–BFR7)
│   ├── fetch_from_db.py    # Retrieves datasets from MySQL (IIR1)
│   ├── embed.py            # Converts records to vectors, stores in ChromaDB (IIR3)
│   └── requirements.txt
├── prompt_generator/       # (WIP) Assembles augmented prompts (IIR2, IIR4)
├── rag_llm/                # (WIP) LLM inference module
├── bounded_context/        # (WIP) Bounded Context output module
├── .env.example            # Environment variable template
├── .gitignore
└── README.md
```
---

## Requirements

- Python 3.11+
- MySQL 8.0+
- Git

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/ragiokay/rag-llm-bounded-context.git
cd rag-llm-bounded-context
````

### 2. Create your `.env` file

```bash
copy .env.example .env
```

Open `.env` and fill in your MySQL credentials:

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=bpc_db
```

### 3. Set up MySQL database

```bash
mysql -u root -p < database/schema.sql
```

### 4. Install dependencies and load dataset into MySQL

```bash
pip install -r database/requirements.txt
python database/seed.py
```

This downloads the BPC dataset from HuggingFace (3,077 rows) and loads it into MySQL.

### 5. Install dependencies and generate vector space

```bash
pip install -r embedding/requirements.txt
python embedding/embed.py
```

This reads from MySQL, converts all records into vectors using `all-MiniLM-L6-v2`, and stores them in ChromaDB under `vector_db/`. 9 collections are created, one per domain.

---

## Datasets

|Dataset|Source|Rows|Domain|
|---|---|---|---|
|BPC|[ibm-research/BPC](https://huggingface.co/datasets/ibm-research/BPC)|3,077|Business Process Causal Reasoning|

---

## ChromaDB Collections

After running `embed.py`, the following collections are available in `vector_db/`:

|Collection|Domain|Vectors|
|---|---|---|
|bpc_education|Education|327|
|bpc_finance|Finance|391|
|bpc_human_resources|Human Resources|381|
|bpc_insurance|Insurance|272|
|bpc_logistics|Logistics|296|
|bpc_manufacturing|Manufacturing|364|
|bpc_medical|Medical|345|
|bpc_retail|Retail|232|
|bpc_transportation|Transportation|469|

---

## Notes

- `vector_db/` is not committed to Git. It is generated locally by running `embed.py`.
- `.env` is not committed to Git. Never commit real credentials.
- MySQL is the source of truth. ChromaDB can always be rebuilt from MySQL.

---

## Requirements Traceability

|Code|Requirement|Implemented In|
|---|---|---|
|BFR1|Schema definition|`database/schema.sql`|
|BFR2|Data storage|`database/seed.py`|
|BFR5|Organize and clean datasets|`embedding/embed.py`|
|BFR6|Transfer datasets into vector space|`embedding/embed.py`|
|BFR7|Review and clean vector space|`embedding/embed.py`|
|IIR1|Database ↔ Embedding Module|`embedding/fetch_from_db.py`|
|IIR3|Embedding Module ↔ Vector Database|`embedding/embed.py`|

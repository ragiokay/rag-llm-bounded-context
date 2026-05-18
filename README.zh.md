# RAG-LLM 有界上下文提取器

透過檢索增強生成（RAG）搭配大型語言模型（LLM），自動化識別有界上下文（Bounded Context）。

---

## 系統流程

```
Qdrant（向量資料庫，已預先 seed 至 server）
↓ retrieve.py：query_similar / query_all
Prompt Generator
↓ 增強後的 prompt
vLLM（Llama-3.2-1B-Instruct）
↓
Bounded Context 輸出（JSON）
```

---

## 給組員：如何使用 Retriever

向量資料庫已預先 seed 完成，直接 import `retrieve.py` 即可使用，不需要本地設定。

### 環境變數（由 Salvador 在共用 container 中設定）

| 變數 | 值 |
|---|---|
| `QDRANT_URL` | `http://140.112.90.146:6333` |
| `COLLECTION_PREFIX` | `spring2026SE_g1_rag_` |

### Import 方式

```python
import sys
sys.path.insert(0, "embedding")   # 依實際路徑調整
from retrieve import query_similar, query_all
```

---

### `query_similar(query_text, collection, n_results)`

查詢單一 collection。

| 參數 | 型別 | 預設值 | 說明 |
|---|---|---|---|
| `query_text` | `str` | 必填 | 自然語言查詢句 |
| `collection` | `str` | `"maven_ere_causal"` | Collection 名稱（含 prefix） |
| `n_results` | `int` | `3` | 回傳筆數 |

```python
results = query_similar(
    "暴風雨造成嚴重洪災。",
    collection="spring2026SE_g1_rag_maven_ere_causal",
    n_results=3,
)
```

---

### `query_all(query_text, n_results_per_collection)`

同時查詢所有 collection，結果依相似度排序。

| 參數 | 型別 | 預設值 | 說明 |
|---|---|---|---|
| `query_text` | `str` | 必填 | 自然語言查詢句 |
| `n_results_per_collection` | `int` | `3` | 每個 collection 回傳筆數 |

```python
results = query_all("乾旱導致糧食短缺。", n_results_per_collection=2)
```

---

### 回傳格式

兩個函式都回傳 `list[dict]`，每筆結構如下：

```json
{
  "input": "被比對到的文件內容",
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

| 欄位 | 說明 |
|---|---|
| `input` | 比對到的文件原文 |
| `distance` | 餘弦距離（0 = 完全相同，1 = 完全無關），越低越相似 |
| `collection` | 結果來自哪個 collection |
| `output.policy` | 因果/命令關係標籤 |
| `output.domain_event` | 原因句或事件描述 |
| `output.command` | 結果句或命令名稱 |
| `output.bounded_context` | 領域 / 來源背景 |
| `output.aggregate` | 聚合根或事件類型 |
| `output.source_phrase` | 原始來源文字 |
| `output.trigger_span` | 命令觸發片語（僅 MTOP） |

---

## 可用 Collection（已預先 seed）

所有 collection 使用前綴 `spring2026SE_g1_rag_`。

| Collection | 資料集 | 向量數 | 說明 |
|---|---|---|---|
| `bpc_education` | BPC | 327 | 業務流程因果推理 — 教育 |
| `bpc_finance` | BPC | 391 | 金融 |
| `bpc_human_resources` | BPC | 381 | 人力資源 |
| `bpc_insurance` | BPC | 272 | 保險 |
| `bpc_logistics` | BPC | 296 | 物流 |
| `bpc_manufacturing` | BPC | 364 | 製造業 |
| `bpc_medical` | BPC | 345 | 醫療 |
| `bpc_retail` | BPC | 232 | 零售 |
| `bpc_transportation` | BPC | 469 | 交通運輸 |
| `maven_ere_causal` | MAVEN-ERE | 11,000+ | 事件因果/前提關係對 |
| `mtop_commands` | MTOP | 11,159 | 任務導向命令語句 |

---

## 專案結構

```
rag-llm-bounded-context/
├── embedding/
│   ├── retrieve.py          # 檢索介面（query_similar, query_all）
│   ├── embed.py             # BPC seed 腳本
│   ├── seed_maven_ere.py    # MAVEN-ERE seed 腳本
│   ├── seed_mtop.py         # MTOP seed 腳本
│   ├── reset_collections.py # 維護工具：列出 / 刪除 collection
│   ├── causal_transform.py  # BPC 資料列 → CausalRelationRecord（DDD schema）
│   └── requirements.txt
├── tests/                   # 162 個單元測試（使用 in-memory Qdrant）
├── Dockerfile
└── README.md
```

---

## 重新 Seed（必要時）

資料已 seed 完成。只有在 server 資料遺失時才需要重跑。

```bash
# 先設環境變數
export QDRANT_URL=http://140.112.90.146:6333
export COLLECTION_PREFIX=spring2026SE_g1_rag_

# BPC（自動從 HuggingFace 下載）
python embedding/embed.py

# MAVEN-ERE（需要 data/maven_ere/train.jsonl，或自動 fallback 至 HuggingFace）
python embedding/seed_maven_ere.py

# MTOP（需要 data/mtop/en/train.txt，從 https://fb.me/mtop_dataset 下載）
python embedding/seed_mtop.py
```

查看或清除 collection：

```bash
python embedding/reset_collections.py --list
python embedding/reset_collections.py --prefix spring2026SE_g1_rag_
```

---

## 執行測試

```bash
pip install -r embedding/requirements.txt
python -m pytest tests/ -q
```

所有測試使用 in-memory Qdrant，不需要連線到 server。

---

## 需求追蹤

| 代碼 | 需求 | 實作位置 |
|---|---|---|
| BFR5 | 整理與清理資料集 | `embedding/embed.py`, `causal_transform.py` |
| BFR6 | 將資料集轉入向量空間 | `embedding/embed.py`, `seed_maven_ere.py`, `seed_mtop.py` |
| BFR7 | 審查與清理向量空間 | `embedding/embed.py`（`review_collection`） |
| IIR3 | Embedding Module ↔ 向量資料庫 | `embedding/retrieve.py` |

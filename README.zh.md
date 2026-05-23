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
from retrieve import query_similar, query_all, query_multiple, query_by_prefix
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

### `query_multiple(query_text, collections, n_results_per_collection)`

查詢指定的 collection 清單，結果依距離排序合併回傳。

| 參數 | 型別 | 預設值 | 說明 |
|---|---|---|---|
| `query_text` | `str` | 必填 | 自然語言查詢句 |
| `collections` | `list[str]` | 必填 | 要查詢的完整 collection 名稱清單 |
| `n_results_per_collection` | `int` | `3` | 每個 collection 回傳筆數 |

```python
results = query_multiple(
    "使用者取消已排定的會議。",
    collections=["spring2026SE_g1_rag_mtop_commands",
                 "spring2026SE_g1_rag_log_commands"],
    n_results_per_collection=3,
)
```

---

### `query_by_prefix(query_text, prefixes, n_results_per_collection)`

搜尋所有名稱符合任一 prefix 的 collection，`COLLECTION_PREFIX` 環境變數會自動加在每個 prefix 前面。

| 參數 | 型別 | 預設值 | 說明 |
|---|---|---|---|
| `query_text` | `str` | 必填 | 自然語言查詢句 |
| `prefixes` | `list[str]` | 必填 | 要比對的 collection 名稱前綴（不含 `COLLECTION_PREFIX`） |
| `n_results_per_collection` | `int` | `3` | 每個 collection 回傳筆數 |

```python
# 搜尋所有 log_* collection（log_domain_events + log_commands）
results = query_by_prefix("會議被發起人取消。", prefixes=["log"])

# 同時搜尋 bpc_* 和 log_* collection
results = query_by_prefix("使用者送出表單。", prefixes=["bpc", "log"])
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
| `maven_ere_causal` | MAVEN-ERE | 36,316 | 事件因果/前提關係對 |
| `mtop_commands` | MTOP | 11,159 | 任務導向命令語句 |
| `log_domain_events` | DDD Log | 85 | 從 DDD 自動提取工具 log 中萃取的 Domain Events |
| `log_commands` | DDD Log | 69 | 從 DDD 自動提取工具 log 中萃取的 Commands（含 Actor） |

---

## 專案結構

```
rag-llm-bounded-context/
├── embedding/
│   ├── retrieve.py          # 檢索介面（query_similar / query_all / query_multiple / query_by_prefix）
│   ├── embed.py             # BPC seed 腳本
│   ├── seed_maven_ere.py    # MAVEN-ERE seed 腳本
│   ├── seed_mtop.py         # MTOP seed 腳本
│   ├── seed_log.py          # DDD Log seed 腳本（domain events + commands）
│   ├── reset_collections.py # 維護工具：列出 / 刪除 collection
│   ├── causal_transform.py  # BPC 資料列 → CausalRelationRecord（DDD schema）
│   └── requirements.txt
├── tests/                   # 202 個單元測試（使用 in-memory Qdrant）
│   ├── test_log.py          # seed_log.py 的 parser 測試
│   └── test_retrieve_log.py # log collection 的 retrieval 測試
├── reports/                 # 自動產生的 demo 輸出（retrieve.py __main__）
│   ├── retrieve_demo_*.md   # 可閱覽的結果表格
│   └── retrieve_demo_*.json # 各 query 的完整 JSON 結果
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

# DDD Log（需要 data/log/*.log — 由 DDD 自動提取工具產生）
python embedding/seed_log.py
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

產生完整 demo 報告（需要 Qdrant 運行中）：

```bash
python embedding/retrieve.py
# 輸出 reports/retrieve_demo_<timestamp>.md  （可閱覽的結果表格）
# 輸出 reports/retrieve_demo_<timestamp>.json（各 query 完整 JSON 結果）
```

---

## 需求追蹤

| 代碼 | 需求 | 實作位置 |
|---|---|---|
| BFR5 | 整理與清理資料集 | `embedding/embed.py`, `causal_transform.py` |
| BFR6 | 將資料集轉入向量空間 | `embedding/embed.py`, `seed_maven_ere.py`, `seed_mtop.py`, `seed_log.py` |
| BFR7 | 審查與清理向量空間 | `embedding/embed.py`（`review_collection`） |
| IIR3 | Embedding Module ↔ 向量資料庫 | `embedding/retrieve.py` |

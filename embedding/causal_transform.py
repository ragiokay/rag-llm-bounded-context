# Causal Relation Transformer Module
# Converts raw BPC dataset rows into DDD-structured CausalRelationRecord objects
# that the Prompt Generator can consume directly without additional parsing.

from pydantic import BaseModel, field_validator


class CausalRelationRecord(BaseModel):
    """
    Interface contract between the Embedding Module and the Prompt Generator.
    Every record stored in ChromaDB must conform to this schema.
    """
    id: str
    cause: str            # causal event or condition (from BPC phrase)
    consequence: str      # outcome or effect (from BPC question, polarity encoded by answer)
    bounded_context: str  # business domain (e.g. Finance, Logistics)
    category: str         # causal category label from BPC (e.g. cause, enable, prevent)
    source_phrase: str    # original sentence kept for traceability
    embed_text: str       # the string that was fed to the embedding model

    @field_validator("cause", "consequence", "bounded_context", "category", "embed_text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty or whitespace")
        return v.strip()

    def to_chroma_metadata(self) -> dict:
        """Returns only the fields that ChromaDB stores as metadata (no embed_text)."""
        return {
            "cause": self.cause,
            "consequence": self.consequence,
            "bounded_context": self.bounded_context,
            "category": self.category,
            "source_phrase": self.source_phrase,
        }


def transform_row(row: dict) -> CausalRelationRecord | None:
    """
    Transforms one raw BPC MySQL row into a CausalRelationRecord.
    Returns None if the row lacks the minimum required fields.

    BPC columns used:
        id       -> record id
        phrase   -> cause (the business situation)
        question -> raw text for consequence extraction
        answer   -> 'yes'/'no' polarity that qualifies the consequence
        category -> causal category label
        domain   -> bounded_context
    """
    def _str(val) -> str:
        return "" if val is None else str(val).strip()

    phrase = _str(row.get("phrase"))
    question = _str(row.get("question"))
    answer = _str(row.get("answer")).lower()
    category = _str(row.get("category"))
    domain = _str(row.get("domain"))

    if not phrase or not question or not domain:
        return None

    # Encode answer polarity into the consequence string so the generator
    # can distinguish confirmed effects from negated ones.
    polarity = "does" if answer == "yes" else "does not"
    consequence = f"{polarity} — {question}"

    embed_text = f"{phrase} {question}"

    try:
        return CausalRelationRecord(
            id=str(row.get("id", "")),
            cause=phrase,
            consequence=consequence,
            bounded_context=domain,
            category=category if category else "unknown",
            source_phrase=phrase,
            embed_text=embed_text,
        )
    except Exception:
        return None


def transform_batch(rows: list[dict]) -> tuple[list[CausalRelationRecord], int]:
    """
    Transforms a list of raw rows. Returns (valid_records, skipped_count).
    """
    records: list[CausalRelationRecord] = []
    skipped = 0

    for row in rows:
        record = transform_row(row)
        if record is None:
            skipped += 1
        else:
            records.append(record)

    return records, skipped

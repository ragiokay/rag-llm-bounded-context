# Causal Relation Transformer Module
# Converts raw BPC dataset rows into Event Storming / DDD-structured records
# that the Prompt Generator can consume directly without additional parsing.

from pydantic import BaseModel, field_validator
from typing import Optional


class CausalRelationRecord(BaseModel):
    """
    Interface contract between the Embedding Module and the Prompt Generator.
    Fields follow Event Storming element names (Brandolini).
    Every record stored in ChromaDB must conform to this schema.

    BPC dataset covers 5 of the 8 Event Storming elements:
        domain_event, command, policy, aggregate, bounded_context.
    The remaining three (views, user_roles, process) require other datasets.
    """
    id: str
    domain_event: str        # business event that occurred (BPC: phrase)
    command: str             # action/query triggered by the event (BPC: question)
    policy: str              # causal rule: "does/does not — {question}" (BPC: answer+question)
    aggregate: str           # causal category grouping (BPC: category)
    bounded_context: str     # business domain (BPC: domain)
    source_phrase: str       # original sentence kept for traceability
    embed_text: str          # string fed to the embedding model
    views: Optional[str] = None        # data views — not available in BPC
    user_roles: Optional[str] = None   # actor — not available in BPC
    process: Optional[str] = None      # process — not available in BPC

    @field_validator("domain_event", "command", "policy", "aggregate", "bounded_context", "embed_text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty or whitespace")
        return v.strip()

    def to_chroma_metadata(self) -> dict:
        """
        Returns fields stored as ChromaDB metadata payload.
        Optional fields are excluded when None so ChromaDB doesn't store nulls.
        """
        meta = {
            "domain_event": self.domain_event,
            "command": self.command,
            "policy": self.policy,
            "aggregate": self.aggregate,
            "bounded_context": self.bounded_context,
            "source_phrase": self.source_phrase,
        }
        if self.views is not None:
            meta["views"] = self.views
        if self.user_roles is not None:
            meta["user_roles"] = self.user_roles
        if self.process is not None:
            meta["process"] = self.process
        return meta


def transform_row(row: dict) -> "CausalRelationRecord | None":
    """
    Transforms one raw BPC MySQL row into a CausalRelationRecord.
    Returns None if the row lacks the minimum required fields.

    BPC column mapping:
        phrase   -> domain_event  (the business situation / event)
        question -> command       (the action or query being triggered)
        answer   -> used to encode polarity in policy
        category -> aggregate     (causal category label)
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

    # Encode answer polarity into the policy string so the generator
    # can distinguish confirmed causal rules from negated ones.
    polarity = "does" if answer == "yes" else "does not"
    policy = f"{polarity} — {question}"

    embed_text = f"{phrase} {question}"

    try:
        return CausalRelationRecord(
            id=str(row.get("id", "")),
            domain_event=phrase,
            command=question,
            policy=policy,
            aggregate=category if category else "unknown",
            bounded_context=domain,
            source_phrase=phrase,
            embed_text=embed_text,
        )
    except Exception:
        return None


def transform_batch(rows: list[dict]) -> "tuple[list[CausalRelationRecord], int]":
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

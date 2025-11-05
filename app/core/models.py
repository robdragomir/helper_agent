from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field


class EvidencePack(BaseModel):
    mode: Literal["offline", "online"]
    context_text: str
    # How confident we think this context actually answers the query (0.0 - 1.0)
    coverage_confidence: float
    # Notes such as "no relevant internal info found" or "sources disagree"
    notes: Optional[str] = None



class FinalAnswer(BaseModel):
    # Text we will show to the CLI user
    text: str

    # Did we use local KB? web? both?
    used_offline: bool
    used_online: bool

    # Final confidence estimate (0.0 - 1.0)
    answer_confidence: float

    # List of normalized citations like:
    # { "label": "[1]", "source": "https://example.com", "note": "online source" }
    citations: List[Dict[str, str]] = Field(default_factory=list)


class KnowledgeBaseSnapshot(BaseModel):
    # Where we download the canonical text file from
    source_url: str
    # Where we store the current local copy of that file
    local_file_path: str

    # Hash of the *current* remote content (the last downloaded version).
    content_hash: str

    # When we last successfully updated from remote
    last_updated_at: datetime

    # Marker for the embeddings / vector index version. Could be timestamp,
    # could be hash of embeddings.
    embedding_index_version: str

    # True means embeddings reflect current text content.
    # False means we detected content drift and haven't rebuilt yet.
    is_fresh: bool


class InferenceTrace(BaseModel):
    # What the user actually asked
    query_text: str

    # The search mode used: offline, online, or both
    route: Literal["offline", "online", "both"]

    # Previews (first N chars) of the evidence we assembled for each path.
    offline_context_preview: Optional[str] = None
    online_context_preview: Optional[str] = None

    # What we actually returned to the CLI
    final_answer: FinalAnswer

    # Perf / cost info
    latency_ms_total: int
    token_usage: Dict[str, int] = Field(default_factory=dict)

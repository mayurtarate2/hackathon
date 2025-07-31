# document_pipeline/chunk_schema.py

from pydantic import BaseModel, Field
from typing import List, Tuple
import json

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique ID for this chunk")
    text: str = Field(..., description="Text content of the chunk")
    embedding: List[float] = Field(..., description="1536-d vector from embedding model")
    token_count: int = Field(..., description="Number of tokens in this chunk")
    char_range: Tuple[int, int] = Field(..., description="Start and end character positions in the document")

    def __str__(self):
        return f"<{self.chunk_id} | {self.token_count} tokens>"

    def to_dict(self, flatten_char_range: bool = False):
        base = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "token_count": self.token_count,
        }

        if flatten_char_range:
            base["char_start"] = self.char_range[0]
            base["char_end"] = self.char_range[1]
        else:
            base["char_range"] = self.char_range

        return base

    def to_json(self, flatten_char_range: bool = False):
        return json.dumps(self.to_dict(flatten_char_range=flatten_char_range))

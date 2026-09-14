"""Public FastAPI request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    chunk_id: str
    document_id: str
    document_title: str
    section_title: str
    version: str
    source_file: str
    evidence: str


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "partial", "not_found"]
    answer: str
    sources: list[SourceInfo]


class HealthResponse(BaseModel):
    status: Literal["ready"]
    ready: Literal[True]
    loaded_chunk_count: int = Field(ge=0)
    embedding_model_id: str
    generation_model_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail

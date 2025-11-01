from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class AnalysisType(str, Enum):
    LEXICAL_STATS = "lexical_stats"
    POS = "pos"
    NER = "ner"
    READABILITY = "readability"
    KEYWORDS = "keywords"
    EMBEDDINGS = "embeddings"
    SENTIMENT = "sentiment"


class WebPageContent(BaseModel):
    url: HttpUrl | str = Field(..., description="Web page URL")
    title: str = Field(..., min_length=1, description="Page title")
    text: str = Field(..., min_length=1, description="Extracted text content")
    html: str = Field(..., min_length=1, description="Raw HTML content")
    timestamp: float = Field(..., description="Extraction timestamp (epoch seconds)")
    word_count: int = Field(..., ge=0, description="Total word count")
    char_count: int = Field(..., ge=0, description="Total character count")
    language: str = Field(default="", description="ISO 639-1 language code")
    language_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Text content too short for meaningful analysis")
        return v.strip()


class LinguisticAnalysisResult(BaseModel):
    analysis_type: AnalysisType
    name: str
    results: dict[str, Any]
    processing_time: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

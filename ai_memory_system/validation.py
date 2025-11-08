"""
Input validation using Pydantic.

Security: Prevents XSS, injection, DoS attacks
References: OWASP Top 10, FastAPI validation
"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import Literal, Optional, Dict, Any


class InteractionRequest(BaseModel):
    """Validated interaction request."""

    type: Literal['chat', 'feedback', 'update', 'identity_update'] = Field(
        ...,
        description="Interaction type"
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Interaction content"
    )

    significance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Significance score [0, 1]"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata"
    )

    @validator('content')
    def sanitize_content(cls, v):
        """Sanitize content to prevent XSS."""
        # Normalize whitespace
        v = ' '.join(v.split())

        # XSS detection
        dangerous = ['<script', 'javascript:', 'onerror=', 'onclick=']
        v_lower = v.lower()
        for pattern in dangerous:
            if pattern in v_lower:
                raise ValueError(f'Suspicious content: {pattern}')

        return v

    class Config:
        schema_extra = {
            "example": {
                "type": "chat",
                "content": "What is AI?",
                "significance": 0.8
            }
        }

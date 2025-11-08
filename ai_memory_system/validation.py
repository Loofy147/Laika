from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class InteractionModel(BaseModel):
    type: str = Field(..., description="The type of interaction, e.g., 'chat' or 'identity_update'.")
    content: str = Field(..., description="The content of the interaction.")
    significance: float = Field(..., ge=0.0, le=1.0, description="The significance of the interaction.")

class IdentityModel(BaseModel):
    interests: Optional[List[str]] = None
    biography: Optional[str] = None
    additional_properties: Optional[Dict[str, Any]] = None

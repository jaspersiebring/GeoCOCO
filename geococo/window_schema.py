from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, root_validator

class WindowSchema(BaseModel):
    width_window: int = Field(gt=0)
    height_window: int = Field(gt=0)
    width_overlap: int = Field(ge=0, default=0)
    height_overlap: int = Field(ge=0, default=0)
    width_step: Optional[int] = Field(gt=0)
    height_step: Optional[int]= Field(gt=0)
    
    @root_validator(pre=True)
    def _calculate_steps(cls: WindowSchema, values: Dict[str, Any]) -> Dict[str, Any]:
        values["width_step"] = values["width_window"] - values["width_overlap"] * 2
        values["height_step"] = values["height_window"] - values["height_overlap"] * 2
        return values
    
# Call update_forward_refs() to resolve forward references (for pydantic <2.0.0)
WindowSchema.update_forward_refs()
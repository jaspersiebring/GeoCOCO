from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class WindowSchema(BaseModel):
    width_window: int = Field(gt=0)
    height_window: int = Field(gt=0)
    width_overlap: int = Field(ge=0, default=0)
    height_overlap: int = Field(ge=0, default=0)
    width_step: Optional[int] = Field(gt=0)
    height_step: Optional[int]= Field(gt=0)
        
    @validator("width_step", "height_step", always=True, pre=True)
    def _calculate_steps(cls: WindowSchema, v: Any, values: Dict[str, Any], field: str) -> Any:
        if field.name == "width_step":
            v = values["width_window"] - values["width_overlap"] * 2
        else:
            v = values["height_window"] - values["height_overlap"] * 2
        return v
    
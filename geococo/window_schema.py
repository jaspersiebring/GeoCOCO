from typing import Optional
from pydantic import model_validator, BaseModel, Field
from typing_extensions import Self

class WindowSchema(BaseModel, validate_assignment=True):
    width_window: int = Field(gt=0, strict=True)
    height_window: int = Field(gt=0, strict=True)
    width_overlap: int = Field(ge=0, default=0, strict=True)
    height_overlap: int = Field(ge=0, default=0, strict=True)
    width_step: Optional[int] = Field(None, gt=0, strict=True)
    height_step: Optional[int] = Field(None, gt=0, strict=True)
    
    @model_validator(mode="after")
    def _calculate_steps(self) -> Self:
        if self.width_step == None and self.height_step == None:
            self.width_step = self.width_window - self.width_overlap * 2
            self.height_step = self.height_window - self.height_overlap * 2
        return self

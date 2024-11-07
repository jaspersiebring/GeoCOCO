from __future__ import annotations
from typing import Optional
from pydantic import model_validator, BaseModel, Field


class StaticSchema(BaseModel):
    width_window: int = Field(gt=0, strict=True)
    height_window: int = Field(gt=0, strict=True)
    width_overlap: int = Field(ge=0, default=0, strict=True)
    height_overlap: int = Field(ge=0, default=0, strict=True)
    width_step: Optional[int] = Field(None, gt=0, strict=True)
    height_step: Optional[int] = Field(None, gt=0, strict=True)


class WindowSchema(StaticSchema):
    @model_validator(mode="after")
    def _calculate_steps(self) -> StaticSchema:
        self.width_step = self.width_window - self.width_overlap * 2
        self.height_step = self.height_window - self.height_overlap * 2
        return StaticSchema(**self.__dict__)

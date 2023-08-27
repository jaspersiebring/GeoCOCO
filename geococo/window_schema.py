from pydantic import BaseModel, Field, StrictInt, model_validator
from typing import Optional


class WindowSchema(BaseModel):
    width_window: StrictInt = Field(gt=0)
    height_window: StrictInt = Field(gt=0)
    width_overlap: StrictInt = Field(ge=0, default=0)
    height_overlap: StrictInt = Field(ge=0, default=0)
    width_step: Optional[StrictInt] = Field(gt=0)
    height_step: Optional[StrictInt]= Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def _calculate_steps(cls, data):
        data["width_step"] = data["width_window"] - data["width_overlap"] * 2
        data["height_step"] = data["height_window"] - data["height_overlap"] * 2
        return data
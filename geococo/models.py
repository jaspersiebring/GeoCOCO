from __future__ import annotations
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, InstanceOf, model_validator
from typing import Optional, List, Dict
from datetime import datetime
import pathlib

@dataclass
class WindowSource:
    width_window_pixels: int = Field(gt=0)
    width_overlap_pixels: int = Field(ge=0)
    height_window_pixels: int = Field(gt=0)
    height_overlap_pixels: int = Field(ge=0)
    
    def __post_init__(self):
        self.width_step_pixels: int = (
            self.width_window_pixels - self.width_overlap_pixels * 2
        )
        self.height_step_pixels: int = (
            self.height_window_pixels - self.height_overlap_pixels * 2
        )

        if self.width_step_pixels <= 0 and self.height_step_pixels <= 0:
            raise ValueError(
                f"Both width_step_pixels ({self.width_step_pixels}) and height_step_pixels ({self.height_step_pixels}) must be positive. Increase width_window_pixels ({self.width_window_pixels}) and height_window_pixels ({self.height_window_pixels}) or decrease width_overlap_pixels ({self.width_overlap_pixels}) and height_overlap_pixels ({self.height_overlap_pixels})"
            )
        if self.width_step_pixels <= 0:
            raise ValueError(
                f"width_step_pixels ({self.width_step_pixels}) must be positive, increase width_window_pixels ({self.width_window_pixels}) or decrease width_overlap_pixels ({self.width_overlap_pixels})"
            )
        if self.height_step_pixels <= 0:
            raise ValueError(
                f"height_step_pixels ({self.height_step_pixels}) must be positive, increase height_window_pixels ({self.height_overlap_pixels}) or decrease height_overlap_pixels ({self.height_overlap_pixels})"
            )


class Info(BaseModel):
    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    url: Optional[AnyUrl] = None
    date_created: Optional[datetime] = None

class Image(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int 
    width: int
    height: int
    file_name: pathlib.Path

class Annotation(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    image_id: int
    category_id: int
    segmentation: Dict
    area: float
    bbox: List[float] 
    iscrowd: int

class Category(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    name: str
    supercategory: str

class CocoDataset(BaseModel):
    info: Info
    images: List[InstanceOf[Image]] = []
    annotations: List[InstanceOf[Annotation]] = []
    categories: List[InstanceOf[Category]] = []
    _next_image_id: int = 1
    _next_annotation_id: int = 1
    
    @model_validator(mode="after")
    def _set_ids(self) -> CocoDataset:
        self._next_image_id = len(self.images) + 1
        self._next_annotation_id = len(self.annotations) + 1
        return self
        
    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)
        self._next_annotation_id += 1

    def add_image(self, image: Image) -> None:
        self.images.append(image)
        self._next_image_id += 1

    @property
    def next_image_id(self) -> int: 
        return self._next_image_id
    
    @property
    def next_annotation_id(self) -> int:
        return self._next_annotation_id
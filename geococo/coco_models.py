from __future__ import annotations
import pathlib
from datetime import datetime
from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import AnyUrl, BaseModel, ConfigDict, InstanceOf, model_validator


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
    segmentation: Segmentation
    area: float
    bbox: List[float]
    iscrowd: int


class Category(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    name: str
    supercategory: str


class Segmentation(TypedDict):
    size: List[int]
    counts: bytes
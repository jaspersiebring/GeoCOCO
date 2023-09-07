from __future__ import annotations
import pathlib
from datetime import datetime
from typing import List, Optional, Set
from typing_extensions import TypedDict
from pydantic import BaseModel, ConfigDict, InstanceOf, model_validator


class CocoDataset(BaseModel):
    info: Info
    images: List[InstanceOf[Image]] = []
    annotations: List[InstanceOf[Annotation]] = []
    categories: List[InstanceOf[Category]] = []
    sources: List[InstanceOf[Source]] = []
    _next_image_id: int = 1
    _next_annotation_id: int = 1
    _next_source_id: int = 1
    _unique_output_dirs: Set[pathlib.Path] = set()
    _unique_source_images : Set[pathlib.Path] = set()
    
    @model_validator(mode="after")
    def _set_ids(self) -> CocoDataset:
        self._next_image_id = len(self.images) + 1
        self._next_annotation_id = len(self.annotations) + 1
        self._next_source_id = len(self.sources) + 1
        return self

    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)
        self._next_annotation_id += 1

    def add_image(self, image: Image) -> None:
        self.images.append(image)
        self._next_image_id += 1

    def add_source(self, source: Source) -> None:
        self.sources.append(source)
        self._next_source_id += 1
    
    @property
    def next_image_id(self) -> int:
        return self._next_image_id

    @property
    def next_annotation_id(self) -> int:
        return self._next_annotation_id
    
    @property
    def next_source_id(self) -> int:
        return self._next_source_id


class Info(BaseModel):
    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    date_created: Optional[datetime] = None


class Image(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    width: int
    height: int
    file_name: pathlib.Path
    source_id: int


class Annotation(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    image_id: int
    category_id: int
    segmentation: RleDict
    area: float
    bbox: List[float]
    iscrowd: int


class Category(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    name: str
    supercategory: str


class RleDict(TypedDict):
    size: List[int]
    counts: bytes


class Source(BaseModel):
    source_id: int
    file_name: pathlib.Path
from __future__ import annotations
import numpy as np
import pathlib
from datetime import datetime
from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, ConfigDict, InstanceOf, model_validator
from semver.version import Version


class CocoDataset(BaseModel):
    info: Info
    images: List[InstanceOf[Image]] = []
    annotations: List[InstanceOf[Annotation]] = []
    categories: List[InstanceOf[Category]] = []
    sources: List[InstanceOf[Source]] = []
    _next_image_id: int = 1
    _next_annotation_id: int = 1
    _next_source_id: int = 1

    @model_validator(mode="after")
    def _set_ids(self) -> CocoDataset:
        self._next_image_id = len(self.images) + 1
        self._next_annotation_id = len(self.annotations) + 1
        self._next_source_id = len(self.sources)
        return self

    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)
        self._next_annotation_id += 1

    def add_image(self, image: Image) -> None:
        self.images.append(image)
        self._next_image_id += 1

    def add_source(self, source_path: pathlib.Path) -> None:
        sources = [ssrc for ssrc in self.sources if ssrc.file_name == source_path]
        if sources:
            assert len(sources) == 1
            source = sources[0]
            self.bump_version(bump_method="patch")
        else:
            source = Source(id=len(self.sources) + 1, file_name=source_path)
            self.sources.append(source)
            self.bump_version(bump_method="minor")

        self._next_source_id = source.id

    def bump_version(self, bump_method: str) -> None:
        bump_methods = ["patch", "minor", "major"]
        version = Version.parse(self.info.version)

        if bump_method not in bump_methods:
            raise ValueError(f"bump_method needs to be one of {bump_methods}")
        elif bump_method == bump_methods[0]:
            version = version.bump_patch()
        elif bump_method == bump_methods[1]:
            version = version.bump_minor()
        else:
            version = version.bump_major()

        self.info.version = str(version)

    def verify_new_output_dir(self, images_dir: pathlib.Path) -> None:
        output_dirs = np.unique([image.file_name.parent for image in self.images])
        if images_dir not in output_dirs:
            self.bump_version(bump_method="major")

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
    version: str = str(Version(major=0))
    year: Optional[int] = None
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
    id: int
    file_name: pathlib.Path

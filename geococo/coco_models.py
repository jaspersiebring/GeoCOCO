from __future__ import annotations
import numpy as np
import pathlib
from datetime import datetime
from typing import List, Optional, Union, Dict
from typing_extensions import TypedDict
from numpy.typing import ArrayLike

from pydantic import BaseModel, ConfigDict, InstanceOf, model_validator
from semver.version import Version
from geococo.utils import assert_valid_categories

class CocoDataset(BaseModel):    
    info: Info
    images: List[InstanceOf[Image]] = []
    annotations: List[InstanceOf[Annotation]] = []
    categories: List[InstanceOf[Category]] = []
    sources: List[InstanceOf[Source]] = []
    _next_image_id: int = 1
    _next_annotation_id: int = 1
    _next_source_id: int = 1
    _category_mapper: Dict = {}

    @model_validator(mode="after")
    def _set_ids(self) -> CocoDataset:
        self._next_image_id = len(self.images) + 1
        self._next_annotation_id = len(self.annotations) + 1
        self._next_source_id = len(self.sources)
        self._category_mapper = self._get_category_mapper()
        return self

    def _get_category_mapper(self) -> Dict:
        category_data = [(category.name, category.id) for category in self.categories]
        category_mapper = dict(category_data) if category_data else {}
        return category_mapper

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

    def add_categories(self, categories: np.ndarray) -> None:
        # checking if categories are castable to str and under a certain size
        categories = assert_valid_categories(categories=np.unique(categories))
        
        # filtering existing categories 
        category_mask = np.isin(categories, list(self._category_mapper.keys()))
        new_categories = categories[~category_mask]

        # generating mapper from new categories
        start = len(self._category_mapper.values()) + 1
        end = start + new_categories.size
        category_dict = dict(zip(new_categories, np.arange(start, end)))

        # instance and append new Category objects to dataset
        for category_name, category_id in category_dict.items():
            category = Category(id = category_id, name = str(category_name), supercategory="1")
            self.categories.append(category)

        # update existing category_mapper with new categories
        self._category_mapper.update(category_dict)

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

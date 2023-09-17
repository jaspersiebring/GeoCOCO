from __future__ import annotations
import numpy as np
import pathlib
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, ConfigDict, root_validator
from pydantic.fields import Field
from semver.version import Version


class CocoDataset(BaseModel):
    info: Info
    images: List[Image] = []
    annotations: List[Annotation] = []
    categories: List[Category] = []
    sources: List[Source] = []
    next_image_id: int = Field(default=1, exclude=True)
    next_annotation_id: int = Field(default=1, exclude=True)
    next_source_id: int = Field(default=1, exclude=True)
    
    @root_validator
    def _set_ids(cls: CocoDataset, values: Dict[str, Any]) -> Dict[str, Any]:
        values["next_image_id"] = len(values["images"]) + 1
        values["next_annotation_id"] = len(values["annotations"]) + 1
        values["next_source_id"] = len(values["sources"])
        return values
        
    def add_annotation(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)
        self.next_annotation_id += 1

    def add_image(self, image: Image) -> None:
        self.images.append(image)
        self.next_image_id += 1

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

        self.next_source_id = source.id
    
    
    def add_categories(self, category_ids: Optional[np.ndarray], category_names: Optional[np.ndarray], supercategory_names: Optional[np.ndarray]) -> None:
        # Loading all existing Category instances as a single dataframe
        category_pd = pd.DataFrame([category.dict() for category in self.categories])
        super_default = "1"

        # creating supercategory_names if not exists
        if not isinstance(supercategory_names, np.ndarray) and isinstance(category_ids, np.ndarray):
            supercategory_names = np.full(shape=category_ids.shape, fill_value=super_default)
        elif not isinstance(supercategory_names, np.ndarray) and isinstance(category_names, np.ndarray):
            supercategory_names = np.full(shape=category_names.shape, fill_value=super_default)
        else:
            raise AttributeError("At least one category attribute must be present")
        
        # i.e. all new and all present
        if isinstance(category_ids, np.ndarray) and isinstance(category_names, np.ndarray):
            assert category_ids.shape == category_names.shape
            id_mask = np.isin(category_ids, category_pd["id"].values)        
            for cid, name, supercategory in zip(category_ids[~id_mask], category_names[~id_mask], supercategory_names[~id_mask]):
                category = Category(id=cid, name=name, supercategory=supercategory)    
                self.categories.append(category)
        elif isinstance(category_ids, np.ndarray):
            #i.e. if only category_ids are given, no names
            id_mask = np.isin(category_ids, category_pd["id"].values)
            
            for cid, name, supercategory in zip(category_ids[~id_mask], category_ids[~id_mask].astype(str), supercategory_names[~id_mask]):
                category = Category(id=cid, name=name, supercategory=supercategory)    
                self.categories.append(category)
        elif isinstance(category_names, np.ndarray):
            #i.e. if only category_names are given, no ids
            name_mask = np.isin(category_names, category_pd["name"].values)

            #new names, arange from latest id    
            start = category_pd.loc[category_pd["name"].isin(category_names[name_mask]), "id"].max() + 1
            end = start + category_names[~name_mask].size
            new_category_ids = np.arange(start, end)
        
            for cid, name, supercategory in zip(new_category_ids, category_names[~name_mask], supercategory_names[~name_mask]):
                category = Category(id=cid, name=name, supercategory=supercategory)    
                self.categories.append(category)
        else:
            raise AttributeError("At least one category attribute must be present")

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


# Call update_forward_refs() to resolve forward references (for pydantic <2.0.0)
CocoDataset.update_forward_refs()
Info.update_forward_refs()
Image.update_forward_refs()
Annotation.update_forward_refs()
Category.update_forward_refs()
Source.update_forward_refs()
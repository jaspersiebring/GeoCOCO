from __future__ import annotations
import numpy as np
import pathlib
import pandas as pd
from pandas import Series
from datetime import datetime
from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, root_validator
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

    def add_categories(
        self,
        category_ids: Optional[Series],
        category_names: Optional[Series],
        super_names: Optional[Series],
    ) -> None:
        # initializing values
        super_default = "1"
        names_present = ids_present = False

        # Loading all existing Category instances as a single dataframe
        category_pd = pd.DataFrame(
            [category.dict() for category in self.categories],
            columns=Category.schema()["properties"].keys(),
        )

        # checking if names can be assigned to uid_array (used to check duplicates)
        if category_names is not None:
            category_names: np.ndarray = category_names.to_numpy()
            uid_array = category_names
            uid_attribute = "name"
            names_present = True

        # checking if ids can be assigned to uid_array (used to check duplicates)
        if category_ids is not None:
            category_ids: np.ndarray = category_ids.to_numpy()
            uid_array = category_ids  # overrides existing array because ids are leading
            uid_attribute = "id"
            ids_present = True
        if not names_present and not ids_present:
            raise AttributeError("At least one category attribute must be present")

        # masking out duplicate values and exiting if all duplicates
        original_shape = uid_array.shape
        _, indices = np.unique(uid_array, return_index=True)
        uid_array = uid_array[indices]
        member_mask = np.isin(uid_array, category_pd[uid_attribute])
        new_members = uid_array[~member_mask]
        new_shape = new_members.shape
        if new_shape[0] == 0:
            return

        # creating default supercategory_names if not given
        if super_names is None:
            super_names = np.full(
                shape=new_shape,
                fill_value=super_default
                ) # type: ignore[assignment]
        else:
            super_names: np.ndarray = super_names.to_numpy()
            assert super_names.shape == original_shape
            super_names = super_names[indices][~member_mask]

        # creating default category_names if not given (str version of ids)
        if ids_present and not names_present:
            category_names = new_members.astype(str)
            category_ids = new_members
        # creating ids if not given (incremental sequence starting from last known id)
        elif names_present and not ids_present:
            pandas_mask = category_pd[uid_attribute].isin(uid_array[member_mask])
            max_id = category_pd.loc[pandas_mask, "id"].max()
            start = np.nansum([max_id, 1])
            end = start + new_members.size
            category_ids = np.arange(start, end) # type: ignore[assignment]
            category_names = new_members
        # ensuring equal size for category names and ids (if given)
        else:
            assert category_names.shape == original_shape  # type: ignore[union-attr]
            category_names = category_names[indices][~member_mask] # type: ignore[index]
            category_ids = new_members

        # iteratively instancing and appending Category from set ids, names and supers
        cip = zip(category_ids, category_names, super_names)
        for cid, name, super in cip:
            category = Category(id=cid, name=name, supercategory=super)
            self.categories.append(category)

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
    id: int
    width: int
    height: int
    file_name: pathlib.Path
    source_id: int


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: RleDict
    area: float
    bbox: List[float]
    iscrowd: int


class Category(BaseModel):
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

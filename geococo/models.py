from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime
import pathlib

@dataclass
class WindowSource:
    width_window_pixels: int
    width_overlap_pixels: int
    height_window_pixels: int
    height_overlap_pixels: int
    
    def __post_init__(self):
        # type checking (maybe just use pydantic?)
        for var_name, var_type in self.__annotations__.items():
            if not isinstance(getattr(self, var_name), var_type):
                raise ValueError(f"self.{var_name} is not of required type {var_type}")

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
    url: Optional[str] = None
    date_created: Optional[datetime] = None

class Image(BaseModel):
    id: int 
    width: int
    height: int
    file_name: pathlib.Path
    date_captured: Optional[datetime] = None

class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: Dict
    area: int
    bbox: List[int] 
    iscrowd: int

class Category(BaseModel):
    id: int
    name: str
    supercategory: str

# add read_from and save_as methods
class CocoDataset(BaseModel):
    info: Info
    images: List[Image]
    annotations: List[Annotation]
    categories: List[Category]
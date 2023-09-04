import os
import numpy as np
import pathlib
from datetime import datetime
from geococo.coco_models import CocoDataset, Info, Image, Annotation, Category, RleDict

"""
Preface: we're mainly testing the logic inside the pydantic models, not the models/fields themselves (that would be a bit redundant)
"""


def test_dataset():
    info = Info(
        year=3030,
        version="1.0",
        description="Test",
        contributor="Test user",
        date_created=datetime.now(),
    )

    dataset = CocoDataset(info=info, images=[], annotations=[], categories=[])
    assert dataset.next_annotation_id == 1
    assert dataset.next_image_id == 1


def test_dataset_add_annotations():
    """Checks annotation_id increment"""

    dataset = CocoDataset(info=Info())
    assert dataset.next_annotation_id == 1
    assert dataset.next_image_id == 1
    n_annotations = np.random.randint(2, 10)
    segmentation = RleDict(size= [256, 256], counts=  os.urandom(np.random.randint(1, 100)))

    for _ in range(n_annotations):
        ann = Annotation(
            id=dataset.next_annotation_id,
            image_id=1,
            category_id=1,
            segmentation=segmentation,
            area=10.0,
            bbox=[1.0, 2.0, 3.0, 4.0],
            iscrowd=0,
        )
        dataset.add_annotation(annotation=ann)

    assert n_annotations == dataset.next_annotation_id - 1
    assert n_annotations == len(dataset.annotations)


def test_dataset_add_images():
    """Checks image_id increment"""

    dataset = CocoDataset(info=Info())
    assert dataset.next_annotation_id == 1
    assert dataset.next_image_id == 1
    n_images = np.random.randint(2, 10)

    for _ in range(n_images):
        img = Image(
            id=dataset.next_image_id,
            width=512,
            height=512,
            file_name=pathlib.Path("image.png"),
        )
        dataset.add_image(image=img)

    assert n_images == dataset.next_image_id - 1
    assert n_images == len(dataset.images)


def test_info():
    """Simple instance test"""

    Info(
        year=None,
        version=None,
        description=None,
        contributor=None,
        date_created=datetime.now(),
    )


def test_image():
    """Simple instance test"""

    Image(id=1, width=512, height=512, file_name=pathlib.Path("image.png"))


def test_annotation():
    """Simple instance test"""

    segmentation = RleDict(size= [256, 256], counts=  os.urandom(np.random.randint(1, 100)))

    Annotation(
        id=1,
        image_id=1,
        category_id=1,
        segmentation=segmentation,
        area=10.0,
        bbox=[1.0, 2.0, 3.0, 4.0],
        iscrowd=0,
    )


def test_category():
    """Simple instance test"""

    Category(id=1, name="", supercategory="")

def test_segmentation():
    """Simple instance test"""
    
    RleDict(size= [256, 256], counts=  os.urandom(np.random.randint(1, 100)))
import os
import numpy as np
import pathlib
from datetime import datetime
from geococo.coco_models import (
    CocoDataset,
    Info,
    Image,
    Annotation,
    Category,
    RleDict,
    Source,
)


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
    """Checks annotation_id increment."""

    dataset = CocoDataset(info=Info())
    assert dataset.next_annotation_id == 1
    assert dataset.next_image_id == 1
    n_annotations = np.random.randint(2, 10)
    segmentation = RleDict(
        size=[256, 256], counts=os.urandom(np.random.randint(1, 100))
    )

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
    """Checks image_id increment."""

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
            source_id=1,
        )
        dataset.add_image(image=img)

    assert n_images == dataset.next_image_id - 1
    assert n_images == len(dataset.images)


def test_info():
    """Simple instance test."""

    Info(
        year=None,
        description=None,
        contributor=None,
        date_created=datetime.now(),
    )


def test_image():
    """Simple instance test."""

    Image(id=1, width=512, height=512, file_name=pathlib.Path("image.png"), source_id=1)


def test_annotation():
    """Simple instance test."""

    segmentation = RleDict(
        size=[256, 256], counts=os.urandom(np.random.randint(1, 100))
    )

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
    """Simple instance test."""

    Category(id=1, name="", supercategory="")


def test_segmentation():
    """Simple instance test."""

    RleDict(size=[256, 256], counts=os.urandom(np.random.randint(1, 100)))


def test_source():
    """Simple instance test."""

    Source(file_name=pathlib.Path(), id=1)


def test_dataset_add_sources():
    """Checks proper incrementation of source_id."""

    # Bit different from the other ids since we check for duplication
    # and only increment if new
    dataset = CocoDataset(info=Info())
    assert dataset.next_source_id == 0
    dataset.add_source(source_path=pathlib.Path("a"))
    assert dataset.next_source_id == 1
    dataset.add_source(source_path=pathlib.Path("a"))
    assert dataset.next_source_id == 1
    dataset.add_source(source_path=pathlib.Path("b"))
    assert dataset.next_source_id == 2


def test_dataset_versions():
    """Checks proper incrementation of dataset versions."""

    dataset = CocoDataset(info=Info())
    assert dataset.info.version == "0.0.0"

    # minor bump if same output_dir but different raster_source
    dataset.add_source(source_path=pathlib.Path("a"))
    assert dataset.info.version == "0.1.0"

    # patch bump: if same output_dir and same raster_source
    dataset.add_source(source_path=pathlib.Path("a"))
    assert dataset.info.version == "0.1.1"

    # major bump: if new output_dir
    dataset.verify_new_output_dir(images_dir=pathlib.Path("b"))
    assert dataset.info.version == "1.0.0"


def test_add_categories():
    """Checks independent mapping of category_attribute to class_ids."""

    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    assert dataset._category_mapper == {}

    # adding three unique classes
    categories = np.array(["A", "B", "B", "E", "E"])
    dataset.add_categories(categories=categories)

    # checking length and sequential category_ids
    assert np.unique(categories).size == len(dataset._category_mapper)
    assert np.unique(categories).size == len(dataset.categories)
    assert np.all(np.diff(list(dataset._category_mapper.values())) == 1)

    # check if existing key value pairs don't change
    # done by adding a bunch of existing classes and one new one
    initial_mapper = dataset._category_mapper.copy()
    categories = np.array(["One", "Two", "Two", "Five", "Five", "Six", "Six"])
    dataset.add_categories(categories=categories)
    subset_mapper = {
        key: value
        for key, value in dataset._category_mapper.items()
        if key in initial_mapper.keys()
    }
    assert initial_mapper == subset_mapper
    assert np.all(np.diff(list(dataset._category_mapper.values())) == 1)

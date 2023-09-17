from __future__ import annotations
import pytest
import os
import numpy as np
import pathlib
from datetime import datetime
import geopandas as gpd
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


def test_add_categories_by_ids():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = None
    supercategory_names = None
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_ids).size
    assert np.array_equal(np.array([cat.id for cat in dataset.categories]), np.unique(category_ids))
    assert np.array_equal(np.array([cat.name for cat in dataset.categories]), np.unique(category_ids).astype(str))
    

def test_add_categories_by_names():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = None
    category_names = np.array([1, 2, 2, 5, 5]).astype(str)
    supercategory_names = None
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_names).size
    assert np.array_equal(np.array([cat.id for cat in dataset.categories]), np.arange(1, np.unique(category_names).size +1))
    assert np.array_equal(np.array([cat.name for cat in dataset.categories]), np.unique(category_names))
    

def test_add_categories_by_names_and_ids():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = np.array(["One", "Two", "Two", "Five", "Five"])  
    supercategory_names = None
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_names).size
    assert len(dataset.categories) == np.unique(category_ids).size
    assert np.all(np.isin(np.unique(category_ids), np.array([cat.id for cat in dataset.categories])))
    assert np.all(np.isin(np.unique(category_names), np.array([cat.name for cat in dataset.categories])))
    


def test_add_categories_with_supers():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = np.array(["One", "Two", "Two", "Five", "Five"])  
    supercategory_names = np.array(["A", "A", "A", "B", "B"])  
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_names).size
    assert len(dataset.categories) == np.unique(category_ids).size
    assert np.all(np.isin(np.unique(category_ids), [cat.id for cat in dataset.categories]))
    assert np.all(np.isin(np.unique(category_names), [cat.name for cat in dataset.categories]))
    assert np.all(np.isin(np.unique(supercategory_names), [cat.supercategory for cat in dataset.categories]))
    

def test_add_categories_by_names_and_ids_append_specific():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = np.array(["One", "Two", "Two", "Five", "Five"])  
    supercategory_names = None
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_names).size
    assert len(dataset.categories) == np.unique(category_ids).size

    assert np.all(np.isin(np.unique(category_ids), np.array([cat.id for cat in dataset.categories])))
    assert np.all(np.isin(np.unique(category_names), np.array([cat.name for cat in dataset.categories])))
    
    n_categories = len(dataset.categories)

    # adding one new class and 4 duplicates
    category_ids = np.array([1, 8, 2, 5, 5])
    category_names = np.array(["One", "Eight", "Two", "Five", "Five"])  
    supercategory_names = None

    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )

    assert len(dataset.categories) == n_categories + 1
    assert dataset.categories[-1].id == 8
    assert dataset.categories[-1].name == "Eight"


def test_add_categories_by_names_and_ids_append_auto():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = np.array(["One", "Two", "Two", "Five", "Five"])  
    supercategory_names = None
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_names).size
    assert len(dataset.categories) == np.unique(category_ids).size

    assert np.all(np.isin(np.unique(category_ids), np.array([cat.id for cat in dataset.categories])))
    assert np.all(np.isin(np.unique(category_names), np.array([cat.name for cat in dataset.categories])))
    
    n_categories = len(dataset.categories)

    # adding one new class and 4 duplicates (only by name, not by id)
    category_ids = None
    category_names = np.array(["One", "Eight", "Two", "Five", "Five"])  
    supercategory_names = None

    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )

    assert len(dataset.categories) == n_categories + 1
    assert dataset.categories[-1].id == 6
    assert dataset.categories[-1].name == "Eight"


def test_add_categories_by_names_and_ids_duplicates():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # adding three unique classes
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = np.array(["One", "Two", "Two", "Five", "Five"])  
    supercategory_names = None
    
    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )
    
    assert len(dataset.categories) == np.unique(category_names).size
    assert len(dataset.categories) == np.unique(category_ids).size

    assert np.all(np.isin(np.unique(category_ids), np.array([cat.id for cat in dataset.categories])))
    assert np.all(np.isin(np.unique(category_names), np.array([cat.name for cat in dataset.categories])))
    
    n_categories = len(dataset.categories)
    cats = [cat for cat in dataset.categories]
    
    # adding duplicate ids and names
    category_ids = np.array([1, 2, 2, 5, 5])
    category_names = np.array(["One", "Two", "Two", "Five", "Five"])  
    supercategory_names = None

    dataset.add_categories(
        category_ids=category_ids,
        category_names=category_names,
        supercategory_names=supercategory_names
        )

    assert len(dataset.categories) == n_categories
    assert np.all([cat == dataset.categories[i] for i, cat in enumerate(cats)])




def test_add_categories_faulty():
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # No ids or names
    category_ids = None
    category_names = None
    supercategory_names = None
    with pytest.raises(AttributeError):
        dataset.add_categories(
            category_ids=category_ids,
            category_names=category_names,
            supercategory_names=supercategory_names
            )
        
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # Ids of different len than names
    category_ids = np.array([1, 2, 3])
    category_names = np.array(["1", "2"])
    with pytest.raises(AssertionError):
        dataset.add_categories(
            category_ids=category_ids,
            category_names=category_names,
            supercategory_names=supercategory_names
            )
    
    dataset = CocoDataset(info=Info())
    assert dataset.categories == []
    
    # Non-array input
    category_ids = [1, 2, 3]
    category_names = ["1", "2", "3"]
    with pytest.raises(AttributeError):
        dataset.add_categories(
            category_ids=category_ids,
            category_names=category_names,
            supercategory_names=supercategory_names
            )
    



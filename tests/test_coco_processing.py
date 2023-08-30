import geopandas as gpd
import rasterio
import numpy as np
from datetime import datetime
from pyproj.crs.crs import CRS
from geococo.coco_processing import labels_to_dataset
from geococo.coco_models import CocoDataset, Info
from tests.fixtures import raster_factory, vector_factory

def test_labels_to_dataset_new_dataset(raster_factory, vector_factory, tmp_path):
    json_path = tmp_path / "dataset.json"
    window_bounds = [(256, 256)]
    class_dict = {1:5, 2:12, 4:20}

    # creating raster/vector with overlapping extent
    vector_path = vector_factory(class_dict, 0, 1000, CRS.from_epsg(3857), 20)
    raster_path = raster_factory(1000, 1000, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    src = rasterio.open(raster_path)
    
    # Creating empty CocoDataset as input for labels_to_dataset
    info = Info(version = "0.0.1", date_created=datetime.now())
    dataset = CocoDataset(info=info)
    dataset = labels_to_dataset(
        dataset = dataset,
        images_dir = tmp_path,
        src = src,
        labels = labels,
        window_bounds = window_bounds
        )
    
    # Checking if output has correct classes
    unique_ann_ids = np.unique([ann.category_id for ann in dataset.annotations])
    assert np.all(np.isin(unique_ann_ids, list(class_dict.keys())))

    # Dumping to JSON    
    dst_json_data = dataset.model_dump_json()
    with open(json_path, 'w') as dst_json_fp:
        dst_json_fp.write(dst_json_data)

    # Reading from JSON to verify correct encoding/decoding
    with open(json_path) as src_json_fp:
        src_json_data = src_json_fp.read()
    
    # Reinstating CocoDataset from json and verifying if still identical
    reinstated_dataset = CocoDataset.model_validate_json(src_json_data)
    assert reinstated_dataset == dataset

def test_labels_to_dataset_append_dataset(raster_factory, vector_factory, tmp_path):
    window_bounds = [(256, 256)]
    class_dict = {1:5, 2:12, 4:20}

    # creating raster/vector with overlapping extent
    vector_path = vector_factory(class_dict, 0, 1000, CRS.from_epsg(3857), 20)
    raster_path = raster_factory(1000, 1000, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    src = rasterio.open(raster_path)
    
    # Creating empty CocoDataset as input for labels_to_dataset
    info = Info(version = "0.0.1", date_created=datetime.now())
    dataset = CocoDataset(info=info)
    dataset = labels_to_dataset(
        dataset = dataset,
        images_dir = tmp_path,
        src = src,
        labels = labels,
        window_bounds = window_bounds
        )
    
    # Checking if output has correct classes
    unique_ann_ids = np.unique([ann.category_id for ann in dataset.annotations])
    assert np.all(np.isin(unique_ann_ids, list(class_dict.keys())))


    # Rerunning with existing CocoDataset to verify append
    previous_dataset = dataset.copy(deep=True)
    previous_image_id = dataset.next_image_id - 1 
    previous_annotation_id = dataset.next_annotation_id - 1
    
    dataset = labels_to_dataset(
        dataset = dataset,
        images_dir = tmp_path,
        src = src,
        labels = labels,
        window_bounds = window_bounds
        )
    
    # Checking whether new data was added without touching existing data
    current_annotations = dataset.annotations[:previous_annotation_id]
    current_images = dataset.images[:previous_image_id]

    assert dataset.next_image_id != previous_image_id
    assert dataset.next_annotation_id != previous_annotation_id
    assert len(current_annotations) == len(previous_dataset.annotations)
    assert len(current_images) == len(previous_dataset.images)    
    assert np.all([current_images[i] == ann for i, ann in enumerate(previous_dataset.images)])
    assert np.all([current_annotations[i] == image for i, image in enumerate(previous_dataset.annotations)])
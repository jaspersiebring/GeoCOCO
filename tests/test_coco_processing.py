import pathlib
import geopandas as gpd
import rasterio
import numpy as np
from datetime import datetime
from geococo.coco_processing import labels_to_dataset
from geococo.coco_models import CocoDataset, Info


def test_labels_to_dataset_new_dataset(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    json_path = tmp_path / "dataset.json"
    category_attribute = "category_id"

    with rasterio.open(test_raster) as raster_source:
        # Creating empty CocoDataset as input for labels_to_dataset
        info = Info(version="0.0.1", date_created=datetime.now())
        dataset = CocoDataset(info=info)
        dataset = labels_to_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            category_attribute=category_attribute
        )

        # Checking if output has correct classes
        dataset_class_names =  np.array([cat.name for cat in dataset.categories])
        labels_class_names = overlapping_labels[category_attribute].unique().astype(str)
        assert np.all(np.isin(dataset_class_names, labels_class_names))

        # Dumping to JSON
        dst_json_data = dataset.model_dump_json()
        with open(json_path, "w") as dst_json_fp:
            dst_json_fp.write(dst_json_data)

        # Reading from JSON to verify correct encoding/decoding
        with open(json_path) as src_json_fp:
            src_json_data = src_json_fp.read()

        # Reinstating CocoDataset from json and verifying if still identical
        reinstated_dataset = CocoDataset.model_validate_json(src_json_data)
        assert reinstated_dataset == dataset


def test_labels_to_dataset_append_dataset(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    category_attribute = "category_id"
    with rasterio.open(test_raster) as raster_source:
        # Creating empty CocoDataset as input for labels_to_dataset
        info = Info(version="0.0.1", date_created=datetime.now())
        dataset = CocoDataset(info=info)
        dataset = labels_to_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            category_attribute=category_attribute
        )

        # Checking if output has correct classes
        dataset_class_names =  np.array([cat.name for cat in dataset.categories])
        labels_class_names = overlapping_labels[category_attribute].unique().astype(str)
        assert np.all(np.isin(dataset_class_names, labels_class_names))

        # Rerunning with existing CocoDataset to verify append
        previous_dataset = dataset.copy(deep=True)
        previous_image_id = dataset.next_image_id - 1
        previous_annotation_id = dataset.next_annotation_id - 1

        dataset = labels_to_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
        )

        # Checking whether new data was added without touching existing data
        current_annotations = dataset.annotations[:previous_annotation_id]
        current_images = dataset.images[:previous_image_id]

        assert dataset.next_image_id != previous_image_id
        assert dataset.next_annotation_id != previous_annotation_id
        assert len(current_annotations) == len(previous_dataset.annotations)
        assert len(current_images) == len(previous_dataset.images)
        assert np.all(
            [current_images[i] == ann for i, ann in enumerate(previous_dataset.images)]
        )
        assert np.all(
            [
                current_annotations[i] == image
                for i, image in enumerate(previous_dataset.annotations)
            ]
        )

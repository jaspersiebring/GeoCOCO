import pathlib
import geopandas as gpd
import rasterio
import numpy as np
from datetime import datetime
from geococo.coco_processing import append_dataset
from geococo.coco_models import CocoDataset, Info


def test_labels_to_dataset_new_dataset(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    json_path = tmp_path / "dataset.json"
    category_attribute = "class_names"

    with rasterio.open(test_raster) as raster_source:
        # Creating empty CocoDataset as input for labels_to_dataset
        info = Info(version="0.0.1", date_created=datetime.now())
        dataset = CocoDataset(info=info)
        dataset = append_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            name_attribute=category_attribute,
        )

        # Checking if output has correct classes
        dataset_class_names = np.array([cat.name for cat in dataset.categories])
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
        reinstated_dataset = CocoDataset.parse_raw(src_json_data)
        assert reinstated_dataset == dataset


def test_labels_to_dataset_append_dataset(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    id_attribute = "category_id"

    with rasterio.open(test_raster) as raster_source:
        # Creating empty CocoDataset as input for labels_to_dataset
        info = Info(version="0.0.1", date_created=datetime.now())
        dataset = CocoDataset(info=info)
        dataset = append_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            id_attribute=id_attribute,
        )

        # Checking if output has correct classes
        dataset_class_names = np.array([cat.name for cat in dataset.categories])
        dataset_class_ids = np.array([cat.id for cat in dataset.categories])
        labels_class_ids = overlapping_labels[id_attribute].unique()
        assert np.all(np.isin(dataset_class_ids, labels_class_ids))
        assert np.all(np.isin(dataset_class_names, labels_class_ids.astype(str)))

        # Rerunning with existing CocoDataset to verify append
        previous_dataset = dataset.model_copy(deep=True)
        previous_image_id = dataset.next_image_id - 1
        previous_annotation_id = dataset.next_annotation_id - 1

        dataset = append_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            id_attribute=id_attribute,
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


def test_labels_to_dataset_append_with_new_categories(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    id_attribute = "category_id"
    name_attribute = "class_names"

    with rasterio.open(test_raster) as raster_source:
        # Creating empty CocoDataset as input for labels_to_dataset
        info = Info(version="0.0.1", date_created=datetime.now())
        dataset = CocoDataset(info=info)
        dataset = append_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            id_attribute=id_attribute,
            name_attribute=name_attribute,
        )

        # Checking if output has correct classes
        dataset_class_names = np.array([cat.name for cat in dataset.categories])
        dataset_class_ids = np.array([cat.id for cat in dataset.categories])
        labels_class_names = overlapping_labels[name_attribute].unique().astype(str)
        labels_class_ids = overlapping_labels[id_attribute].unique()
        assert np.all(np.isin(dataset_class_ids, labels_class_ids))
        assert np.all(np.isin(dataset_class_names, labels_class_names))

        # Rerunning with existing CocoDataset to verify append
        previous_dataset = dataset.model_copy(deep=True)
        previous_image_id = dataset.next_image_id - 1
        previous_annotation_id = dataset.next_annotation_id - 1

        # Changing one entry
        overlapping_labels.loc[2, "category_id"] = 6
        overlapping_labels.loc[2, "class_names"] = "Six"

        dataset = append_dataset(
            dataset=dataset,
            images_dir=tmp_path,
            src=raster_source,
            labels=overlapping_labels,
            window_bounds=[(256, 256)],
            id_attribute=id_attribute,
            name_attribute=name_attribute,
        )

        # Checking whether new data was added without touching existing data
        current_annotations = dataset.annotations[:previous_annotation_id]
        current_images = dataset.images[:previous_image_id]

        # Checking creation of Category with continued index
        assert dataset.categories[-1].name == "Six"
        assert dataset.categories[-1].id == previous_dataset.categories[-1].id + 1
        assert len(dataset.categories) == len(previous_dataset.categories) + 1

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

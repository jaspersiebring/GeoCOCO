import pytest
import geopandas as gpd
from datetime import datetime
import rasterio
import pathlib
from geococo.coco_processing import labels_to_dataset
from geococo.coco_models import Info, CocoDataset
from geococo.coco_manager import load_dataset, create_dataset, save_dataset


def test_load_dataset_full(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    json_path = tmp_path / "dataset.json"

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
            id_attribute="category_id",
        )

    json_data = dataset.json()
    with open(json_path, "w") as dst:
        dst.write(json_data)

    loaded_dataset = load_dataset(json_path=json_path)
    assert loaded_dataset == dataset


def test_load_dataset_empty(tmp_path: pathlib.Path) -> None:
    json_path = tmp_path / "dataset.json"
    info = Info(version="0.0.1", date_created=datetime.now())
    dataset = CocoDataset(info=info)

    json_data = dataset.json()
    with open(json_path, "w") as dst:
        dst.write(json_data)

    loaded_dataset = load_dataset(json_path=json_path)
    assert loaded_dataset == dataset


@pytest.mark.parametrize(
    "contributor, version, description, date_created",
    [
        ("User", "0.1.0", "Test", datetime.now()),
        ("", "123.123.123", "", datetime.now()),
        (None, "0.0.0", None, datetime.now()),
    ],
)
def test_create_dataset(contributor, version, description, date_created) -> None:
    # pretty much already tested by tests/test_coco_models.py
    dataset = create_dataset(
        contributor=contributor,
        version=version,
        date_created=date_created,
        description=description,
    )
    assert isinstance(dataset, CocoDataset)
    assert dataset.next_annotation_id == 1
    assert dataset.next_image_id == 1


def test_save_dataset_empty(tmp_path: pathlib.Path) -> None:
    json_path = tmp_path / "dataset.json"
    info = Info(version="0.0.1", date_created=datetime.now())
    dataset = CocoDataset(info=info)
    save_dataset(dataset=dataset, json_path=json_path)
    loaded_dataset = load_dataset(json_path=json_path)
    assert loaded_dataset == dataset


def test_save_dataset_full(
    tmp_path: pathlib.Path,
    test_raster: pathlib.Path,
    overlapping_labels: gpd.GeoDataFrame,
) -> None:
    json_path = tmp_path / "dataset.json"

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
        )

    save_dataset(dataset=dataset, json_path=json_path)
    loaded_dataset = load_dataset(json_path=json_path)
    assert loaded_dataset == dataset

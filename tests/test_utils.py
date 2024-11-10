import pytest
from datetime import datetime
import pathlib
import rasterio
import numpy as np
from rasterio.windows import Window
from shapely.geometry import Polygon
from geococo.utils import (
    window_intersect,
    generate_window_offsets,
    window_factory,
    reshape_image,
    generate_window_polygon,
    estimate_average_bounds,
    estimate_schema,
    mask_label,
    validate_labels,
    update_labels,
    get_date_created,
)
from geococo.coco_models import Category
from geococo.window_schema import WindowSchema
from pandera.errors import SchemaError
import geopandas as gpd
from string import ascii_lowercase
from typing import Tuple


def test_mask_label_overlapping(
    overlapping_labels: gpd.GeoDataFrame, test_raster: pathlib.Path
) -> None:
    with rasterio.open(test_raster) as raster_source:
        for _, row in overlapping_labels.iterrows():
            masked_label = mask_label(input_raster=raster_source, label=row.geometry)
            assert masked_label.ndim == 2
            assert masked_label.dtype == bool
            assert masked_label.sum() >= row.geometry.area


def test_mask_label_nonoverlapping(
    nonoverlapping_labels: gpd.GeoDataFrame, test_raster: pathlib.Path
) -> None:
    with rasterio.open(test_raster) as raster_source:
        for _, row in nonoverlapping_labels.iterrows():
            masked_label = mask_label(input_raster=raster_source, label=row.geometry)
            assert masked_label.ndim == 2
            assert masked_label.dtype == bool
            assert masked_label.sum() == 0


def test_mask_label_closed(
    overlapping_labels: gpd.GeoDataFrame, test_raster: pathlib.Path
) -> None:
    raster_source = rasterio.open(test_raster)
    raster_source.close()
    with pytest.raises(ValueError):
        mask_label(
            input_raster=raster_source, label=overlapping_labels.geometry.values[0]
        )


def test_window_intersect(
    overlapping_labels: gpd.GeoDataFrame, test_raster: pathlib.Path
) -> None:
    with rasterio.open(test_raster) as raster_source:
        # outer polygons have diameter of 2 so resulting window
        # is 82x82 and offset is 9 (10-2/2)
        window = window_intersect(
            input_raster=raster_source, input_vector=overlapping_labels
        )
        assert window.col_off == 9
        assert window.row_off == 9
        assert window.width == 82
        assert window.height == 82


def test_window_intersect_invalid(
    nonoverlapping_labels: gpd.GeoDataFrame, test_raster: pathlib.Path
) -> None:
    with rasterio.open(test_raster) as raster_source:
        with pytest.raises(ValueError):
            window_intersect(
                input_raster=raster_source, input_vector=nonoverlapping_labels
            )


@pytest.mark.parametrize(
    "padding_value,original_shape,requested_shape",
    [(0, (3, 256, 256), (3, 512, 512)), (-10000, (3, 256, 256), (4, 512, 512))],
)
def test_reshape_image_pad(
    padding_value: int,
    original_shape: Tuple[int, int, int],
    requested_shape: Tuple[int, int, int],
) -> None:
    img_array = np.random.randint(0, 255, original_shape)
    reshaped_image = reshape_image(
        img_array=img_array, shape=requested_shape, padding_value=padding_value
    )
    assert reshaped_image.shape == requested_shape
    assert np.array_equal(
        reshaped_image[: original_shape[0], : original_shape[1], : original_shape[2]],
        img_array,
    )
    assert np.all(
        reshaped_image[:, original_shape[1] :, original_shape[2] :] == padding_value
    )


@pytest.mark.parametrize(
    "original_shape,requested_shape",
    [((3, 256, 256), (3, 200, 210)), ((3, 256, 256), (1, 100, 120))],
)
def test_reshape_image_slicing(
    original_shape: Tuple[int, int, int], requested_shape: Tuple[int, int, int]
) -> None:
    img_array = np.random.randint(0, 255, original_shape)
    reshaped_image = reshape_image(img_array=img_array, shape=requested_shape)
    assert reshaped_image.shape == requested_shape
    assert np.array_equal(
        img_array[: requested_shape[0], : requested_shape[1], : requested_shape[2]],
        reshaped_image,
    )


@pytest.mark.parametrize("span, offset", [(100, 0), (5, -100), (100, 100), (256, 800)])
def test_generate_window_polygon(
    span: int, offset: int, test_raster: pathlib.Path
) -> None:
    with rasterio.open(test_raster) as raster_source:  # only provides transform
        window_geom = generate_window_polygon(
            raster_source, Window(offset, offset, span, span)
        )
        # window_geom.bounds (minx, miny, maxx, maxy)
        assert window_geom.bounds == (offset, -span - offset, span + offset, -offset)


def test_generate_window_polygon_negative(test_raster: pathlib.Path) -> None:
    with rasterio.open(test_raster) as raster_source:  # only provides transform
        with pytest.raises(ValueError):
            generate_window_polygon(raster_source, Window(0, 0, -256, -256))


@pytest.mark.parametrize(
    "quantile, expected_bounds", [(0.1, 2), (0.5, 4), (0.6, 4.8), (0.8, 6.4)]
)
def test_estimate_average_bounds(
    quantile: float, expected_bounds: float, overlapping_labels: gpd.GeoDataFrame
) -> None:
    average_width, average_height = estimate_average_bounds(
        gdf=overlapping_labels, quantile=quantile
    )
    assert average_height == expected_bounds
    assert average_width == expected_bounds


@pytest.mark.parametrize(
    "quantile, bounds", [(0.8, (100, 100)), (0.8, (15, 15)), (0.2, (15, 15))]
)
def test_estimate_schema(
    quantile: float,
    bounds: Tuple[int, int],
    overlapping_labels: gpd.GeoDataFrame,
    test_raster: pathlib.Path,
) -> None:
    with rasterio.open(test_raster) as raster_source:
        schema = estimate_schema(
            gdf=overlapping_labels,
            src=raster_source,
            window_bounds=[bounds],
            quantile=quantile,
        )
        # verifying that bounds are respected (ultimately determines output image size)
        assert schema.width_window == bounds[0]
        assert schema.height_window == bounds[1]

        label_span, _ = estimate_average_bounds(
            gdf=overlapping_labels, quantile=quantile
        )
        # verifying that the label sample fits in the window overlap
        assert schema.width_overlap >= label_span
        assert schema.height_overlap >= label_span


@pytest.mark.parametrize(
    "quantile, bounds", [(0.8, (14, 14)), (0.5, (8, 8)), (0.2, (4, 4))]
)
def test_estimate_schema_invalid(
    quantile: float,
    bounds: Tuple[int, int],
    overlapping_labels: gpd.GeoDataFrame,
    test_raster: pathlib.Path,
) -> None:
    with rasterio.open(test_raster) as raster_source:
        with pytest.raises(ValueError):
            estimate_schema(
                gdf=overlapping_labels,
                src=raster_source,
                window_bounds=[bounds],
                quantile=quantile,
            )


@pytest.mark.parametrize(
    "col_off, row_off, width, height",
    [
        (0, 0, 256, 256),
        (0, 0, 256, 120),
        (100, 100, 256, 256),
        (-100, -100, 256, 256),
        (0, 0, 50, 50),
    ],
)
def test_generate_window_offsets_window(
    col_off: int, row_off: int, width: int, height: int
) -> None:
    schema = WindowSchema(
        width_window=100, width_overlap=10, height_window=100, height_overlap=10
    )  # type: ignore

    window = Window(col_off, row_off, width, height)
    offsets = generate_window_offsets(window=window, schema=schema)
    assert np.all(offsets >= 0)
    assert np.all(offsets[:, 0] <= width + schema.width_window)
    assert np.all(offsets[:, 1] <= height + schema.height_window)


@pytest.mark.parametrize(
    "width_window, width_overlap, height_window, height_overlap",
    [(100, 0, 100, 0), (100, 49, 100, 49), (300, 0, 300, 0)],
)
def test_generate_window_offsets_schema(
    width_window: int, width_overlap: int, height_window: int, height_overlap: int
) -> None:
    window = Window(0, 0, 256, 256)
    schema = WindowSchema(
        width_window=width_window,
        width_overlap=width_overlap,
        height_window=height_window,
        height_overlap=height_overlap,
    )  # type: ignore

    offsets = generate_window_offsets(window=window, schema=schema)
    assert np.all(offsets >= 0)
    assert np.all(offsets[:, 0] <= window.width + schema.width_window)
    assert np.all(offsets[:, 1] <= window.height + schema.height_window)


def test_window_factory_not_boundless() -> None:
    boundless = False
    schema = WindowSchema(
        width_window=100, width_overlap=10, height_window=100, height_overlap=10
    )  # type: ignore
    window = Window(0, 0, 1000, 1000)

    window_extents = np.array(
        [
            [
                child_window.col_off + child_window.width,
                child_window.row_off + child_window.height,
            ]
            for child_window in window_factory(
                parent_window=window, schema=schema, boundless=boundless
            )
        ]
    )
    assert np.all(window_extents[:, 0] <= window.width)
    assert np.all(window_extents[:, 1] <= window.height)


def test_window_factory_boundless() -> None:
    boundless = True
    schema = WindowSchema(
        width_window=100, width_overlap=10, height_window=100, height_overlap=10
    )  # type: ignore
    window = Window(0, 0, 1000, 1000)

    window_extents = np.array(
        [
            [
                child_window.col_off + child_window.width,
                child_window.row_off + child_window.height,
            ]
            for child_window in window_factory(
                parent_window=window, schema=schema, boundless=boundless
            )
        ]
    )
    assert np.any(window_extents[:, 0] >= window.width)
    assert np.any(window_extents[:, 1] >= window.height)


def test_validate_labels(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()
    category_ids = np.arange(1, labels.index.size + 1)
    category_names = np.random.choice(list(ascii_lowercase), labels.index.size)
    supercategory_names = np.random.choice(list(ascii_lowercase), labels.index.size)

    category_id_col = "category_ids"
    category_name_col = "category_names"
    supercategory_col = "supercategory_names"

    labels[category_id_col] = category_ids
    labels[category_name_col] = category_names
    labels[supercategory_col] = supercategory_names

    validated_labels = validate_labels(
        labels=labels,
        id_attribute=category_id_col,
        name_attribute=category_name_col,
        super_attribute=supercategory_col,
    )

    assert validated_labels.shape == labels.shape
    assert np.all(validated_labels == labels)


def test_validate_labels_only_ids(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()
    category_ids = np.arange(1, labels.index.size + 1)
    category_id_col = "category_ids"
    labels[category_id_col] = category_ids

    validated_labels = validate_labels(labels=labels, id_attribute=category_id_col)

    assert validated_labels.shape == labels.shape
    assert np.all(validated_labels == labels)


def test_validate_labels_id_casting(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()
    category_ids = np.arange(1, labels.index.size + 1).astype(float)
    category_id_col = "category_ids"
    labels[category_id_col] = category_ids

    validated_labels = validate_labels(labels=labels, id_attribute=category_id_col)

    assert validated_labels.shape == labels.shape
    assert np.all(validated_labels == labels)
    assert labels[category_id_col].dtype == float
    assert validated_labels[category_id_col].dtype == np.int64


def test_validate_labels_only_names(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()
    category_names = np.random.choice(list(ascii_lowercase), labels.index.size)
    category_name_col = "category_names"
    labels[category_name_col] = category_names
    validated_labels = validate_labels(labels=labels, name_attribute=category_name_col)

    assert validated_labels.shape == labels.shape
    assert np.all(validated_labels == labels)


def test_validate_labels_invalid_geom(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()
    invalid_polygon = Polygon([(0, 0), (2, 0), (1, 1), (1, -1), (0, 0)])
    labels.loc[0, "geometry"] = invalid_polygon

    category_ids = np.arange(1, labels.index.size + 1)
    category_names = np.random.choice(list(ascii_lowercase), labels.index.size)
    supercategory_names = np.random.choice(list(ascii_lowercase), labels.index.size)

    category_id_col = "category_ids"
    category_name_col = "category_names"
    supercategory_col = "supercategory_names"

    labels[category_id_col] = category_ids
    labels[category_name_col] = category_names
    labels[supercategory_col] = supercategory_names

    with pytest.raises(SchemaError):
        _ = validate_labels(
            labels=labels,
            id_attribute=category_id_col,
            name_attribute=category_name_col,
            super_attribute=supercategory_col,
        )


def test_validate_labels_invalid_range(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()

    category_ids = np.arange(labels.index.size)
    category_names = np.random.choice(list(ascii_lowercase), labels.index.size)
    supercategory_names = np.random.choice(list(ascii_lowercase), labels.index.size)

    category_id_col = "category_ids"
    category_name_col = "category_names"
    supercategory_col = "supercategory_names"

    labels[category_id_col] = category_ids
    labels[category_name_col] = category_names
    labels[supercategory_col] = supercategory_names

    with pytest.raises(SchemaError):
        _ = validate_labels(
            labels=labels,
            id_attribute=category_id_col,
            name_attribute=category_name_col,
            super_attribute=supercategory_col,
        )


def test_validate_labels_invalid_type(overlapping_labels: gpd.GeoDataFrame):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()

    category_ids = np.arange(1, labels.index.size + 1)
    category_names = category_ids
    category_id_col = "category_ids"
    category_name_col = "category_names"
    labels[category_id_col] = category_ids
    labels[category_name_col] = category_names

    with pytest.raises(SchemaError):
        _ = validate_labels(
            labels=labels,
            id_attribute=category_id_col,
            name_attribute=category_name_col,
        )


def test_update_labels(overlapping_labels):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()

    # can be any name, as long as they contain validated values (see validate_labels)
    category_id_col = "ids"
    category_name_col = "names"

    # categories are always made from existing or new labels
    ids = np.arange(1, 10)  # i.e. unique
    names = ids.astype(str)
    supers = np.full(names.shape, fill_value="1")
    categories = []
    for cid, name, super in zip(ids, names, supers):
        category = Category(id=cid, name=name, supercategory=super)
        categories.append(category)

    # populating labels
    labels[category_id_col] = np.random.choice(ids, labels.index.size)
    labels[category_name_col] = np.random.choice(names, labels.index.size)

    # adding COCO keys
    updated_labels = update_labels(
        labels=labels,
        categories=categories,
        id_attribute=category_id_col,
        name_attribute=category_name_col,
    )

    # checking if all coco keys are present
    assert np.all(np.isin(["id", "name", "supercategory"], updated_labels.columns))
    assert np.all(np.isin(updated_labels["id"].values, ids))
    assert np.all(np.isin(updated_labels["name"].values, names))

    # shape should not change, only values
    assert labels.shape == updated_labels.shape


def test_update_labels_ids(overlapping_labels):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()

    # can be any name, as long as they contain validated values (see validate_labels)
    category_id_col = "ids"
    category_name_col = None

    # categories are always made from existing or new labels
    ids = np.arange(1, 10)  # i.e. unique
    names = ids.astype(str)
    supers = np.full(names.shape, fill_value="1")
    categories = []
    for cid, name, super in zip(ids, names, supers):
        category = Category(id=cid, name=name, supercategory=super)
        categories.append(category)

    # populating labels
    labels[category_id_col] = np.random.choice(ids, labels.index.size)

    # adding COCO keys
    updated_labels = update_labels(
        labels=labels,
        categories=categories,
        id_attribute=category_id_col,
        name_attribute=category_name_col,
    )

    # checking if all coco keys are present
    assert np.all(np.isin(["id", "name", "supercategory"], updated_labels.columns))
    assert np.all(np.isin(updated_labels["id"].values, ids))
    assert np.all(np.isin(updated_labels["name"].values, names))

    # shape should not change, only values
    assert labels.shape == updated_labels.shape


def test_update_labels_names(overlapping_labels):
    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()

    # can be any name, as long as they contain validated values (see validate_labels)
    category_id_col = None
    category_name_col = "names"

    # categories are always made from existing or new labels
    ids = np.arange(1, 10)  # i.e. unique
    names = ids.astype(str)
    supers = np.full(names.shape, fill_value="1")
    categories = []
    for cid, name, super in zip(ids, names, supers):
        category = Category(id=cid, name=name, supercategory=super)
        categories.append(category)

    # populating labels
    labels[category_id_col] = np.random.choice(ids, labels.index.size)

    # adding COCO keys
    updated_labels = update_labels(
        labels=labels,
        categories=categories,
        id_attribute=category_id_col,
        name_attribute=category_name_col,
    )

    # checking if all coco keys are present
    assert np.all(np.isin(["id", "name", "supercategory"], updated_labels.columns))
    assert np.all(np.isin(updated_labels["id"].values, ids))
    assert np.all(np.isin(updated_labels["name"].values, names))

    # shape should not change, only values
    assert labels.shape == updated_labels.shape


def test_update_labels_faulty(overlapping_labels):
    # only testing missing input and input length (input data is already guaranteed to
    # be valid by other methods)

    # dropping everything except geometry
    labels = overlapping_labels[["geometry"]].copy()

    # can be any name, as long as they contain validated values (see validate_labels)
    category_id_col = "ids"
    category_name_col = "names"

    # categories are always made from existing or new labels
    ids = np.arange(1, 10)  # i.e. unique
    names = ids.astype(str)
    supers = np.full(names.shape, fill_value="1")
    categories = []
    for cid, name, super in zip(ids, names, supers):
        category = Category(id=cid, name=name, supercategory=super)
        categories.append(category)

    # Missing atrributes
    with pytest.raises(AttributeError):
        _ = update_labels(
            labels=labels,
            categories=categories,
            id_attribute=None,
            name_attribute=None,
        )

    # Empty arrays
    labels[category_id_col] = np.empty(shape=labels.index.size)
    labels[category_name_col] = np.empty(shape=labels.index.size)

    with pytest.raises(ValueError):
        _ = update_labels(
            labels=labels,
            categories=categories,
            id_attribute=category_id_col,
            name_attribute=category_name_col,
        )


def test_get_date_created(test_raster: pathlib.Path) -> None:
    # no tags, only last_modified
    with rasterio.open(test_raster) as raster_source:
        date_created = get_date_created(raster_source=raster_source)
        # Get the current datetime with a small tolerance window
        assert np.isclose(date_created.timestamp(), datetime.now().timestamp(), atol=5)

    # tags with parsing (i.e. format mismatch)
    with rasterio.open(test_raster) as raster_source:
        date_string = "2023-09-21T00:50:10.131719"
        mock_tags = {"TIFFTAG_DATETIME": date_string}
        raster_source.tags = lambda: mock_tags
        date_created = get_date_created(raster_source=raster_source)
        assert datetime(2023, 9, 21, 0, 50, 10, 131719) == date_created

    # tags without parsing (matching format)
    with rasterio.open(test_raster) as raster_source:
        date_string = "2022-03-12 13:45:00"  # YYYY:MM:DD
        mock_tags = {"TIFFTAG_DATETIME": date_string}
        raster_source.tags = lambda: mock_tags
        date_created = get_date_created(raster_source=raster_source)
        assert datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") == date_created

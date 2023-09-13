import pytest
import pathlib
import rasterio
import numpy as np
from rasterio.windows import Window
from geococo.utils import (
    window_intersect,
    generate_window_offsets,
    window_factory,
    reshape_image,
    generate_window_polygon,
    estimate_average_bounds,
    estimate_schema,
    mask_label,
    assert_valid_categories
)
from geococo.window_schema import WindowSchema
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



def test_assert_valid_categories() -> None:
    # almost all python objects can be represented by str so we just try casting and verify char length
    category_lengths = [10, 49, 50]
    random_words = [ "".join(np.random.choice(list(ascii_lowercase), cl))  for cl in category_lengths]
    random_words = np.array(random_words)
    
    _ = assert_valid_categories(random_words)

    # float64 
    random_numbers = np.random.randn(3).astype(np.float64)
    _ = assert_valid_categories(random_numbers)

    # longer than <U50
    category_lengths = [51, 70, 120]
    random_words = [ "".join(np.random.choice(list(ascii_lowercase), cl))  for cl in category_lengths]
    random_words = np.array(random_words)

    with pytest.raises(ValueError):
        _ = assert_valid_categories(random_words)

    



    
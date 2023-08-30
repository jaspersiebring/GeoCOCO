import pytest
import rasterio
import numpy as np
from rasterio.windows import Window
from geococo.utils import window_intersect, generate_window_offsets, window_factory, reshape_image, generate_window_polygon, estimate_average_bounds, estimate_schema, mask_label
from geococo.window_schema import WindowSchema
from tests.fixtures import raster_factory, vector_factory
import geopandas as gpd
from pyproj.crs.crs import CRS

@pytest.mark.parametrize("test_input", [1, 5, 20])
def test_mask_label_valid(test_input, raster_factory, vector_factory):
    n_labels = test_input
    vector_path = vector_factory({1:n_labels}, 0, 256, CRS.from_epsg(3857), 1)
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    raster_source = rasterio.open(raster_path)
    
    # shapes can be given one at a time or as multipolygon
    label_mask = mask_label(input_raster=raster_source, label=labels.unary_union)
    # depending on # touching pixels, area can be between 1 and 9 pixels
    assert 1 <= label_mask.sum() <= 9 * n_labels

def test_mask_label_invalid(raster_factory, vector_factory):
    # creating raster/vector with non-overlapping extent
    vector_path = vector_factory({1:1}, 512, 769, CRS.from_epsg(3857), 1)
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    raster_source = rasterio.open(raster_path)
    label_mask = mask_label(input_raster=raster_source, label=labels.geometry[0])
    assert label_mask.sum() == 0

def test_window_intersect_valid(raster_factory, vector_factory):
    # creating raster/vector with overlapping extent
    vector_path = vector_factory({1:5, 2:12}, 0, 256, CRS.from_epsg(3857), 1)
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    
    labels = gpd.read_file(vector_path)
    with rasterio.open(raster_path) as src:
        window_intersect(input_raster=src, input_vector=labels)

def test_window_intersect_invalid(raster_factory, vector_factory):
    # creating raster/vector with non-overlapping extent
    vector_path = vector_factory({1:5, 2:12}, 512, 769, CRS.from_epsg(3857), 1)
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    raster_source = rasterio.open(raster_path)

    with pytest.raises(ValueError):
        window_intersect(input_raster=raster_source, input_vector=labels)

@pytest.mark.parametrize("original_shape,requested_shape", [ ((3, 256, 256), (3, 512, 512)), ((3, 256, 256), (4, 512, 512))])
def test_reshape_image_pad(original_shape, requested_shape):
    img_array = np.random.randint(0, 255, original_shape)
    reshaped_image = reshape_image(img_array=img_array, shape=requested_shape)
    assert reshaped_image.shape == requested_shape
    assert np.array_equal(reshaped_image[:original_shape[0], :original_shape[1], :original_shape[2]], img_array)
    
@pytest.mark.parametrize("original_shape,requested_shape", [ ((3, 256, 256), (3, 200, 210)), ((3, 256, 256), (1, 100, 120))])
def test_reshape_image_slicing(original_shape,requested_shape):
    img_array = np.random.randint(0, 255, original_shape)
    reshaped_image = reshape_image(img_array=img_array, shape=requested_shape)

    assert reshaped_image.shape == requested_shape
    assert np.array_equal( img_array[:requested_shape[0], :requested_shape[1], :requested_shape[2]], reshaped_image)

@pytest.mark.parametrize("width,height", [(256, 256), (256, 100), (800, 800)])
def test_generate_window_polygon(width, height, raster_factory):
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    src = rasterio.open(raster_path)
    window_geom = generate_window_polygon(src, Window(0, 0, width, height))
    assert window_geom.bounds == (0, -height, width, 0)

def test_generate_window_polygon_invalid(raster_factory):
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    src = rasterio.open(raster_path)
    with pytest.raises(ValueError):
        window_geom = generate_window_polygon(src, Window(0, 0, -256, -256))
    
@pytest.mark.parametrize("radius", [1, 5, 20])
def test_estimate_average_bounds(radius, vector_factory):
    radius = 5
    vector_path = vector_factory({1:5, 2:12}, 0, 256, CRS.from_epsg(3857), radius)
    labels = gpd.read_file(vector_path)
    average_width, average_height = estimate_average_bounds(gdf=labels, quantile=0.9)
    assert average_height == radius * 2
    assert average_width == radius * 2

@pytest.mark.parametrize("radius,bounds",  [(10, (41, 41)),(1, (5, 5))])
def test_estimate_schema(radius, bounds, vector_factory, raster_factory):
    # creating raster/vector with overlapping extent
    vector_path = vector_factory({1:5, 2:12}, 0, 256, CRS.from_epsg(3857), radius)
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    raster_source = rasterio.open(raster_path)
    estimate_schema(gdf=labels, src=raster_source, window_bounds=[bounds])

@pytest.mark.parametrize("radius,bounds",  [(10, (40, 40)),(1, (4, 4))])
def test_estimate_schema_invalid(radius, bounds, vector_factory, raster_factory):
    # creating raster/vector with overlapping extent
    vector_path = vector_factory({1:5, 2:12}, 0, 256, CRS.from_epsg(3857), radius)
    raster_path = raster_factory(256, 256, "uint8", CRS.from_epsg(3857))
    labels = gpd.read_file(vector_path)
    raster_source = rasterio.open(raster_path)
    with pytest.raises(ValueError):
        estimate_schema(gdf=labels, src=raster_source, window_bounds=[bounds])

def test_generate_window_offsets():
    width = height = 1000
    width_window = height_window = 100
    width_overlap = height_overlap = 20

    schema = WindowSchema(width_window=width_window, width_overlap=width_overlap, height_window=height_window, height_overlap=height_overlap)
    window = Window(0,0, width, height)
    offsets = generate_window_offsets(window=window, schema=schema)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= (window.width + width_window))
    assert np.all(offsets[:, 0] >= 0)
    assert np.all(offsets[:, 1] <= (window.height + height_window))
    assert np.all(offsets[:, 1] >= 0)

def test_window_factory_not_boundless():
    # checks that 'boundless' clips properly
    width = height = 1000
    width_window = height_window = 100
    width_overlap = height_overlap = 20
    boundless = False
    schema = WindowSchema(width_window=width_window, width_overlap=width_overlap, height_window=height_window, height_overlap=height_overlap)
    window = Window(0,0, width, height)
    window_extents = np.array([[child_window.col_off + child_window.width, child_window.row_off + child_window.height] for child_window in window_factory(parent_window=window, schema=schema, boundless=boundless)])
    assert np.all(window_extents[:, 0] <= window.width)
    assert np.all(window_extents[:, 1] <= window.height)
    
def test_window_factory_boundless():
    width = height = 1000
    width_window = height_window = 100
    width_overlap = height_overlap = 20
    boundless = True
    schema = WindowSchema(width_window=width_window, width_overlap=width_overlap, height_window=height_window, height_overlap=height_overlap)
    window = Window(0,0, width, height)
    window_extents = np.array([[child_window.col_off + child_window.width, child_window.row_off + child_window.height] for child_window in window_factory(parent_window=window, schema=schema, boundless=boundless)])
    assert np.any(window_extents[:, 0] >= window.width)
    assert np.any(window_extents[:, 1] >= window.height)
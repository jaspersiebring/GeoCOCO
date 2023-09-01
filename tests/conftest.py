import geopandas as gpd
import pathlib
import uuid
from typing import Callable, Dict, Tuple
import numpy as np
import pytest
import rasterio
from pyproj.crs.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry.polygon import Point


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    np.random.seed(0)


@pytest.fixture
def raster_factory(
    tmp_path: pathlib.Path,
) -> Callable[[int, int, int, str, CRS, int], pathlib.Path]:
    """
    Pytest "factory" fixture that returns a method that can generate a raster file of a
    given specification.

    :param tmp_path: pytest fixture that gives a temporary directory
    :return: method for generating raster files
    """

    def _generate_raster(
        width: int,
        height: int,
        count: int,
        dtype: str = "uint8",
        crs: CRS = CRS.from_epsg(3857),
        nodata: int = 0,
    ) -> pathlib.Path:
        """
        Generates a random raster file of (3, width, height).astype(dtype) at 0,0 of a given CRS
        and returns a tempdir path to this file. The units of height/weight is determined by the
        CRS (i.e. meters for projected, degrees for geographic).

        :param width: number of pixels in x axis
        :param height: number of pixels in y axis
        :param count: number of bands in generated raster
        :param dtype: dtype for the generated raster
        :param crs: coordinate reference system to use (also determines the pixel_size units)
        :param nodata: what value to use as NULL value in generated raster
        :return: tmp_path to generated raster
        """

        raster_path = tmp_path / f"{uuid.uuid1()}.tif"
        transform = from_origin(0, 0, 1, 1)
        data = np.random.randint(0, 255, [count, width, height])
        data = np.add(data, np.random.randn(*data.shape))
        data = data.astype(dtype)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
        }

        with rasterio.open(raster_path, "w", **profile) as dst:
            dst.write(data)
        return raster_path

    return _generate_raster


@pytest.fixture
def vector_factory(
    tmp_path: pathlib.Path,
) -> Callable[[Dict[int, int], int, int, CRS, int], pathlib.Path]:
    """
    Pytest "factory" fixture that returns a method that can generate a vector file of a
    given specification.

    :param tmp_path: pytest fixture that gives a temporary directory
    :return: method for generating vector files
    """

    def _generate_vector(
        class_dict: Dict[int, int],
        low: int = 0,
        high: int = 256,
        crs: CRS = CRS.from_epsg(3857),
        buffer_size: int = 1,
    ) -> pathlib.Path:
        """

        Randomly generates buffered points between 'low' and 'high' in a given CRS as a
        geopackage and returns a tempdir path to this file. The number of points and their
        associated category_id (i.e. class) is determined by the class_dict dictionary
        ({class_id: n}; e.g. {1:5} or 5 instances of '1'). The CRS determines the unit
        (degree with geographic and meter with projected).

        :param class_dict: the counts of all unique category_ids (e.g. {1:5} or 5 instances of '1')
        :param low: lower bound of point generator
        :param high: upper bound of point generator
        :param crs: coordinate reference system to use (determines location of generated points)
        :param buffer_size: distance to use for point buffer (unit is determined by CRS)
        :return: tmp_path to generated geopackage file
        """

        vector_path = tmp_path / f"{uuid.uuid1()}.gpkg"
        classes = np.repeat(
            a=list(class_dict.keys()), repeats=list(class_dict.values())
        )
        random_points = np.random.randint(low, high, (sum(class_dict.values()), 2))
        random_points = np.add(random_points, np.random.randn(*random_points.shape))
        random_points[:, 1] = -random_points[:, 1]  # i.e. pixel coordinates
        random_Points = [Point(random_point) for random_point in random_points]
        random_gpd = gpd.GeoDataFrame(
            geometry=random_Points, data={"category_id": classes}, crs=crs
        )
        random_gpd.geometry = random_gpd.buffer(distance=buffer_size)
        random_gpd.to_file(vector_path)

        return vector_path

    return _generate_vector


@pytest.fixture
def get_vector(vector_factory) -> pathlib.Path:
    """
    Creates a geopackage, saves it to a temporary path and returns this path.

    :param vector_factory: fixture that generate a vector file of given specification
    :return: path to generated geopackage
    """

    crs = CRS.from_epsg(3857)
    class_dict = {1: 5, 2: 12, 4: 20}  # i.e. 5x'1', 12x'2' and 20x'4'
    span = 1000
    buffer_distance = 20

    return vector_factory(class_dict, 0, span, crs, buffer_distance)


@pytest.fixture
def get_raster(raster_factory) -> pathlib.Path:
    """
    Creates a geotiff, saves it to a temporary path and returns this path.

    :param raster_factory: fixture that generate a raster file of given specification
    :return: path to generated geotiff
    """

    crs = CRS.from_epsg(3857)
    span = 1000
    n_bands = 3
    nodata = 0

    return raster_factory(span, span, n_bands, "uint8", crs, nodata)


@pytest.fixture
def get_overlapping_input(
    raster_factory, vector_factory
) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Creates a geotiff and geopackage with overlapping extents and returns their temp_paths.

    :param raster_factory: fixture that generate a raster file of given specification
    :param vector_factory: fixture that generate a vector file of given specification

    :return: temp_paths to the generated vector and raster files
    """

    crs = CRS.from_epsg(3857)
    class_dict = {1: 5, 2: 12, 4: 20}  # i.e. 5x'1', 12x'2' and 20x'4'
    span = 1000
    n_bands = 3
    nodata = 0
    buffer_distance = 20

    vector_path = vector_factory(class_dict, 0, span, crs, buffer_distance)
    raster_path = raster_factory(span, span, n_bands, "uint8", crs, nodata)

    return raster_path, vector_path


@pytest.fixture
def get_nonoverlapping_input(
    raster_factory, vector_factory
) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Creates a geotiff and geopackage with nonoverlapping extents and returns their temp_paths.

    :param raster_factory: fixture that generate a raster file of given specification
    :param vector_factory: fixture that generate a vector file of given specification

    :return: temp_paths to the generated vector and raster files
    """

    crs = CRS.from_epsg(3857)
    class_dict = {1: 5, 2: 12, 4: 20}  # i.e. 5x'1', 12x'2' and 20x'4'
    span = 1000
    n_bands = 3
    nodata = 0
    buffer_distance = 20

    vector_path = vector_factory(class_dict, 2000, 3000, crs, buffer_distance)
    raster_path = raster_factory(span, span, n_bands, "uint8", crs, nodata)
    return raster_path, vector_path
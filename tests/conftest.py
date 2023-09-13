import geopandas as gpd
import pathlib
import uuid
from typing import Callable
import numpy as np
import pytest
import rasterio
from pyproj.crs.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import Point


@pytest.fixture(scope="session", autouse=True)
def set_seed() -> None:
    np.random.seed(0)


@pytest.fixture
def raster_factory(
    tmp_path: pathlib.Path,
) -> Callable[[int, int, int, str, CRS, int], pathlib.Path]:
    """Pytest "factory" fixture that returns a method that can generate a raster file of
    a given specification.

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
        """Generates a random raster file of (3, width, height).astype(dtype) at 0,0 of
        a given CRS and returns a tempdir path to this file. The units of height/weight
        is determined by the CRS (i.e. meters for projected, degrees for geographic).

        :param width: number of pixels in x axis
        :param height: number of pixels in y axis
        :param count: number of bands in generated raster
        :param dtype: dtype for the generated raster
        :param crs: coordinate reference system to use (also determines the pixel_size
            units)
        :param nodata: what value to use as NULL value in generated raster
        :return: tmp_path to generated raster
        """

        raster_path = tmp_path / f"{uuid.uuid1()}.tif"
        transform = from_origin(0, 0, 1, 1)  # i.e. top left with pixel size 1
        data = np.multiply(np.random.rand(count, width, height), 256).astype(dtype)

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


@pytest.fixture(scope="session")
def overlapping_labels() -> gpd.GeoDataFrame:
    """Pytest fixture that returns hardcoded labels as a geodataframe for testing
    purposes."""

    crs = CRS.from_epsg(3857)
    classes = [1, 2, 2, 5, 5]
    class_names = ["One", "Two", "Two", "Five", "Five"]

    points = [
        Point(10, -10),
        Point(30, -30),
        Point(50, -50),
        Point(70, -70),
        Point(90, -90),
    ]
    buffers = [1, 2, 3, 4, 1]
    polygons = [p.buffer(distance=buffers[i]) for i, p in enumerate(points)]

    labels = gpd.GeoDataFrame(
        geometry=polygons,
        data={"category_id": classes, "class_names": class_names},
        crs=crs,
    )  # type: ignore
    return labels


@pytest.fixture(scope="session")
def nonoverlapping_labels() -> gpd.GeoDataFrame:
    """Pytest fixture that returns hardcoded labels as a geodataframe for testing
    purposes."""

    crs = CRS.from_epsg(3857)
    classes = [1, 2, 2, 5, 5]
    class_names = ["One", "Two", "Two", "Five", "Five"]
    points = [
        Point(510, -510),
        Point(530, -530),
        Point(550, -550),
        Point(570, -570),
        Point(590, -590),
    ]
    buffers = [1, 2, 3, 4, 1]
    polygons = [p.buffer(distance=buffers[i]) for i, p in enumerate(points)]

    labels = gpd.GeoDataFrame(
        geometry=polygons,
        data={"category_id": classes, "class_names": class_names},
        crs=crs,
    )  # type: ignore
    return labels


@pytest.fixture
def test_raster(
    raster_factory: Callable[[int, int, int, str, CRS, int], pathlib.Path]
) -> pathlib.Path:
    """Pytest fixture that generates a geotiff with given specifications, saves it to a
    temporary directory and returns its path.

    :param raster_factory: fixture that generate a raster file of given specification
    :return: path to generated geotiff
    """
    crs = CRS.from_epsg(3857)
    n_bands = 3
    nodata = 0
    width = height = 256
    dtype = "uint8"
    return raster_factory(width, height, n_bands, dtype, crs, nodata)

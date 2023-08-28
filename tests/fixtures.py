import geopandas as gpd
import pathlib
import uuid
from typing import Callable, List
import numpy as np
import pytest
import rasterio
from pyproj.crs.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry.polygon import Point

@pytest.fixture
def raster_factory(tmp_path: pathlib.Path) -> Callable[[int, int, int, str, float], pathlib.Path]:
    def _generate_raster(width: int, height: int, count: int, dtype: str = "uint8", nodata: float= 0) -> pathlib.Path:
        raster_path = tmp_path / f"{uuid.uuid1()}.tif"
        data = np.random.randint(0, 255, (count, height, width)).astype(dtype)
        transform = from_origin(0, 0, 1, 1)      
        crs = CRS.from_epsg(4326)    

        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'nodata': nodata,
        }

        with rasterio.open(raster_path, 'w', **profile) as dst:
            dst.write(data)
        return raster_path
    return _generate_raster


@pytest.fixture
def vector_factory(tmp_path: pathlib.Path) -> Callable[[List[int], List[int], int], pathlib.Path]:
    def _generate_vector(bounds: List[int], classes: List[int], buffer_size: int = 1) -> pathlib.Path:
        vector_path = tmp_path / f"{uuid.uuid1()}.gpkg"
        crs = CRS.from_epsg(4326)
        minx, miny, maxx, maxy = bounds
        x = np.random.randint(minx, maxx, len(classes))
        y = np.random.randint(miny, maxy, len(classes))
        random_points = [Point(xy) for xy in zip(x, y)]
        random_polygons = [random_point.buffer(buffer_size) for random_point in random_points]
        data = {"category_id" : classes, "geometry": random_polygons}
        random_gdf = gpd.GeoDataFrame(data=data, crs=crs)
        random_gdf.to_file(vector_path)  

        return vector_path
    return _generate_vector




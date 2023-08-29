import geopandas as gpd
import pathlib
import uuid
from typing import Callable, Dict
import numpy as np
import pytest
import rasterio
from pyproj.crs.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry.polygon import Point


@pytest.fixture
def raster_factory(tmp_path: pathlib.Path) -> Callable[[int, int, str, CRS], pathlib.Path]:
    """
    Pytest "factory" fixture that returns a method that can generate a raster file of a 
    given specification.
    
    :param tmp_path: pytest fixture that gives a temporary directory
    :return: method for generating raster files
    """    

    def _generate_raster(width: int, height: int, dtype: str = "uint8", crs: CRS = CRS.from_epsg(3857)) -> pathlib.Path:
        """
        Generates a random raster file of (3, width, height).astype(dtype) at 0,0 of a given CRS 
        and returns a tempdir path to this file. The units of height/weight is determined by the
        CRS (i.e. meters for projected, degrees for geographic).

        :param width: number of pixels in x axis
        :param height: number of pixels in y axis
        :param dtype: dtype for the generated raster
        :param crs: coordinate reference system to use (also determines the pixel_size units)
        :return: tmp_path to generated raster
        """
                
        raster_path = tmp_path / f"{uuid.uuid1()}.tif"
        transform = from_origin(0, 0, 1, 1)
        data = np.random.randint(0, 255, [3, width, height])
        data = np.add(data, np.random.randn(*data.shape))
        data = data.astype(dtype)

        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 3,
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'nodata': 0,
        }

        with rasterio.open(raster_path, 'w', **profile) as dst:
            dst.write(data)
        return raster_path
    return _generate_raster


@pytest.fixture
def vector_factory(tmp_path: pathlib.Path) -> Callable[[Dict[int, int], int, int, CRS, int], pathlib.Path]:
    """
    Pytest "factory" fixture that returns a method that can generate a vector file of a 
    given specification.

    :param tmp_path: pytest fixture that gives a temporary directory
    :return: method for generating vector files
    """
    def _generate_vector(class_dict: Dict[int, int], low: int = 0, high: int = 256, crs: CRS = CRS.from_epsg(3857), buffer_size: int = 1) -> pathlib.Path:
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
        classes = np.repeat(a = list(class_dict.keys()), repeats=list(class_dict.values()))
        random_points = np.random.randint(low, high,  (sum(class_dict.values()), 2))
        random_points = np.add(random_points, np.random.randn(*random_points.shape))
        random_points[:, 1] = -random_points[:, 1] #i.e. pixel coordinates
        random_Points = [Point(random_point) for random_point in random_points]
        random_gpd = gpd.GeoDataFrame(geometry=random_Points, data = {'category_id': classes}, crs=crs)
        random_gpd.geometry = random_gpd.buffer(distance=buffer_size)
        random_gpd.to_file(vector_path)

        return vector_path
    
    return _generate_vector

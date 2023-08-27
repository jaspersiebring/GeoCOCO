from typing import Generator, List, Optional, Tuple, Dict
import geopandas as gpd
import numpy as np
from rasterio.io import DatasetReader
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
from geococo.window_schema import WindowSchema
from numpy.ma import MaskedArray
from pycocotools import mask as cocomask


def process_label_mask(label_mask: MaskedArray) -> Tuple[Dict, List[int], int]:
    """
    Generates all necessary values for a COCO Annotation

    :param label_mask: A binary mask for an intersected label
    :return: RLE-encoded annotations and their bounds/total area
    """

    label_mask = np.all(label_mask.mask, axis=0)
    label_mask = np.invert(label_mask)
    rle = cocomask.encode(np.asfortranarray(label_mask))
    bounding_box = cv2.boundingRect(label_mask.astype(np.uint8))
    area = np.sum(label_mask)

    return rle, bounding_box, area


def reshape_image(img_array: np.ndarray, shape: Tuple[int, int, int], padding_value: int = 0) -> np.ndarray:
    """
    Reshapes numpy array to match given shape, done through slicing or padding. 

    :param img_array: the numpy array to be reshaped 
    :param shape: the desired shape (bands, rows, cols)
    :param padding_value: what value to pad img_array with (if too small)
    :return: numpy array in desired shape 
    """

    if len(img_array.shape) != len(shape):
        raise ValueError(f"Number of dimensions have to match ({img_array.shape} != {shape})")
    
    img_array = img_array[:shape[0], :shape[1], :shape[2]]
    img_pads = [(0, max(0, n - img_array.shape[i])) for i, n in enumerate(shape)]
    img_array = np.pad(img_array, img_pads, mode='constant', constant_values=padding_value)
    
    return img_array


def generate_window_polygon(datasource: DatasetReader, window: Window) -> Polygon:
    """
    Turns the spatial bounds of a given window to a shapely.Polygon 
    object in a given dataset's CRS.
    
    :param datasource: a rasterio DatasetReader object that provides the affine transformation
    :param window: bounds to represent as Polygon
    :return: shapely Polygon representing the spatial bounds of a given window in a given CRS
    """

    window_transform = datasource.window_transform(window)
    window_bounds = array_bounds(window.height, window.width, window_transform)
    window_poly = box(*window_bounds)
    return window_poly


def generate_window_offsets(window: Window, schema: WindowSchema) -> np.ndarray:
    """
    Computes an array of window offsets bound by a given window.

    :param window: the bounding window (i.e. offsets will be within its bounds)
    :param schema: the parameters for the window generator
    :return: an array of window offsets within the bounds of window
    """

    col_range = np.arange(
        np.max((0, window.col_off - schema.width_overlap_pixels)),
        window.width + window.col_off - schema.width_overlap_pixels,
        schema.width_step_pixels
    )
    row_range = np.arange(
        np.max((0, window.row_off - schema.height_overlap_pixels)),
        window.height + window.row_off - schema.height_overlap_pixels,
        schema.height_step_pixels
    )

    window_offsets = np.array(np.meshgrid(col_range, row_range))
    window_offsets = window_offsets.T.reshape(-1, 2)

    return window_offsets


def window_factory(parent_window: Window, schema: WindowSchema, boundless: bool = True) -> Generator[Window, None, None]:
    """
    Generator that produces rasterio.Window objects in predetermined steps, within the given Window.

    :param parent_window: the window that provides the bounds for all child_window objects
    :param schema: the parameters that determine the window steps
    :param boundless: whether the child_window should be clipped by the parent_window or not
    :yield: a rasterio.Window used for windowed reading/writing
    """
    
    window_offsets = generate_window_offsets(window=parent_window, schema=schema)
    
    for col_off, row_off in window_offsets:
        child_window = Window(
            col_off=int(col_off),
            row_off=int(row_off),
            width=int(schema.width_window_pixels),
            height=int(schema.height_window_pixels),
        )
        if not boundless:
            child_window = child_window.intersection(parent_window)

        yield child_window

def estimate_average_bounds(gdf: gpd.GeoDataFrame, quantile : float = 0.9) -> Tuple[float, float]:
    """
    Estimates the average size of all features in a GeoDataFrame.
    
    :param gdf: GeoDataFrame that contains all features (i.e. shapely.Geometry objects)
    :param quantile: what quantile will represent the feature population  
    :return: a tuple of floats representing average width and height
    """

    label_bounds = gdf.bounds
    label_widths = label_bounds.apply(lambda row: row['maxx'] - row['minx'], axis=1)
    label_heights = label_bounds.apply(lambda row: row['maxy'] - row['miny'], axis=1)
    average_width = np.nanquantile(label_widths, quantile).astype(float)
    average_height = np.nanquantile(label_heights, quantile).astype(float)
    
    return average_width, average_height

def estimate_schema(gdf: gpd.GeoDataFrame, src: DatasetReader, quantile : float = 0.9, window_bounds:  List[Tuple[int, int]] = [(256, 256), (512, 512)]) -> Optional[WindowSchema]:
    """
    Attempts to find a schema that is able to represent the average GeoDataFrame 
    feature (i.e. sufficient overlap) but within the bounds given by window_bounds.

    :param gdf: GeoDataFrame that contains features that determine the degree of overlap
    :param src: The rasterio DataSource associated with the resulting schema (i.e. bounds and pixelsizes)
    :param quantile: what quantile will represent the feature population  
    :param window_bounds: a list of possible limits for the window generators
    :return: (if found) a viable WindowSchema with sufficient overlap within the window_bounds
    """    

    # estimating the required overlap between windows for labels to be represented fully
    average_width, average_height = estimate_average_bounds(gdf=gdf, quantile=quantile)
    width_overlap_pixels = int(np.ceil(average_width / src.res[0]))
    height_overlap_pixels = int(np.ceil(average_height / src.res[1]))

    schema = None
    last_exception = None
    for i in range(len(window_bounds)):
        width_pixels, height_pixels = window_bounds[i]
        try:
            schema = WindowSchema(
                width_window_pixels=width_pixels,
                width_overlap_pixels=width_overlap_pixels,
                height_window_pixels=height_pixels,
                height_overlap_pixels=height_overlap_pixels)
            break  # Break the loop as soon as a valid schema is found
        
        except ValueError as ve:
            last_exception = ve
            continue
        
    if schema is None:
        raise ValueError(f"No WindowSchema objects could be created from the given window_bounds {window_bounds}, raising last Exception..") from last_exception

    return schema
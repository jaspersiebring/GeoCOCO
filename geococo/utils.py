from rasterio.io import DatasetReader
import geopandas as gpd
from geococo.models import WindowSource
from rasterio.windows import Window
import numpy as np
from typing import Generator, Tuple, Optional, List


def generate_window_offsets(window: Window, window_source: WindowSource) -> np.ndarray:
    """
    Computes an array of window offsets bound by a given window.

    :param window: the bounding window (i.e. offsets will be within its bounds)
    :param window_source: the parameters for the window generator
    :return: an array of window offsets within the bounds of window
    """

    col_range = np.arange(
        max(0, window.col_off - window_source.width_overlap_pixels),
        window.width + window.col_off - window_source.width_overlap_pixels,
        window_source.width_step_pixels
    )
    row_range = np.arange(
        max(0, window.row_off - window_source.height_overlap_pixels),
        window.height + window.row_off - window_source.height_overlap_pixels,
        window_source.height_step_pixels
    )

    window_offsets = np.array(np.meshgrid(col_range, row_range))
    window_offsets = window_offsets.T.reshape(-1, 2)

    return window_offsets


def window_factory(parent_window: Window, window_source: WindowSource, boundless: bool = False) -> Generator[Window, None, None]:
    """
    Generator that produces rasterio.Window objects in predetermined steps, within the given Window.

    :param parent_window: the window that provides the bounds for all child_window objects
    :param window_source: the parameters that determine the window steps
    :param boundless: whether the child_window should be clipped by the parent_window or not
    :yield: a rasterio.Window used for windowed reading/writing
    """
    
    window_offsets = generate_window_offsets(window=parent_window, window_source=window_source)
    
    for col_off, row_off in window_offsets:
        child_window = Window(
            col_off=col_off,
            row_off=row_off,
            width=window_source.width_window_pixels,
            height=window_source.height_window_pixels,
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

def estimate_window_source(gdf: gpd.GeoDataFrame, src: DatasetReader, quantile : float = 0.9, window_bounds:  List[Tuple[int, int]] = [(256, 256), (512, 512)]) -> Optional[WindowSource]:
    """
    Attempts to find a window_source that is able to represent the average GeoDataFrame 
    feature (i.e. sufficient overlap) but within the bounds given by window_bounds.

    :param gdf: GeoDataFrame that contains features that determine the degree of overlap
    :param src: The rasterio DataSource associated with the resulting window_source (i.e. bounds and pixelsizes)
    :param quantile: what quantile will represent the feature population  
    :param window_bounds: a list of possible limits for the window generators
    :return: (if found) a viable WindowSource with sufficient overlap within the window_bounds
    """    

    # estimating the required overlap between windows for labels to be represented fully
    average_width, average_height = estimate_average_bounds(gdf=gdf, quantile=quantile)
    width_overlap_pixels = int(np.ceil(average_width / src.res[0]))
    height_overlap_pixels = int(np.ceil(average_height / src.res[1]))

    window_source = None
    last_exception = None
    for i in range(len(window_bounds)):
        width_pixels, height_pixels = window_bounds[i]
        try:
            window_source = WindowSource(
                width_window_pixels=width_pixels,
                width_overlap_pixels=width_overlap_pixels,
                height_window_pixels=height_pixels,
                height_overlap_pixels=height_overlap_pixels)
        except ValueError as ve:
            last_exception = ve
            continue
        
    if window_source is None:
        raise ValueError(f"No WindowSource objects could be created from the given window_bounds {window_bounds}, raising last Exception..") from last_exception

    return window_source 

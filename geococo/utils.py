from typing import Generator, List, Tuple, Union
import geopandas as gpd
import numpy as np
from rasterio.io import DatasetReader
from rasterio.transform import array_bounds
from rasterio.windows import Window, from_bounds
from rasterio.errors import WindowError
from rasterio.mask import mask as riomask
from shapely.geometry import MultiPolygon, Polygon, box
from geococo.window_schema import WindowSchema


def mask_label(
    input_raster: DatasetReader, label: Union[Polygon, MultiPolygon]
) -> np.ndarray:
    """Masks out an label from input_raster and flattens it to a 2D binary array. If it
    doesn't overlap, the resulting mask will only consist of False bools.

    :param input_raster: open rasterio DatasetReader for the input raster
    :param label: Polygon object representing the area to be masked (i.e. label)
    :return: A 2D binary array representing the label
    """

    if input_raster.closed:
        raise ValueError("Attempted mask with a closed DatasetReader")

    label_mask, _ = riomask(
        dataset=input_raster, shapes=[label], all_touched=True, filled=False
    )
    label_mask = np.all(label_mask.mask, axis=0)
    label_mask = np.invert(label_mask)

    return label_mask


def window_intersect(
    input_raster: DatasetReader, input_vector: gpd.GeoDataFrame
) -> Window:
    """Generates a Rasterio Window from the intersecting extents of the input data. It
    also verifies if the input data share the same CRS and if they physically overlap.

    :param input_raster: rasterio dataset (i.e. input image)
    :param input_vector: geopandas geodataframe (i.e. input labels)
    :return: rasterio window that represent the intersection between input data extents
    """

    if input_vector.crs != input_raster.crs:
        raise ValueError(
            f"CRS of input raster ({input_raster.crs.to_epsg()}) and "
            "labels ({input_vector.crs.to_epsg()}) don't match, exiting.."
        )

    raster_bounds = input_raster.bounds
    vector_bounds = input_vector.total_bounds
    raster_window = from_bounds(*raster_bounds, transform=input_raster.transform)
    vector_window = from_bounds(*vector_bounds, transform=input_raster.transform)

    try:
        intersection_window = vector_window.intersection(raster_window)
    except WindowError as window_error:
        raise ValueError(
            "Extent of input raster and vector don't overlap"
        ) from window_error

    return intersection_window


def reshape_image(
    img_array: np.ndarray, shape: Tuple[int, int, int], padding_value: int = 0
) -> np.ndarray:
    """Reshapes 3D numpy array to match given 3D shape, done through slicing or padding.

    :param img_array: the numpy array to be reshaped
    :param shape: the desired shape (bands, rows, cols)
    :param padding_value: what value to pad img_array with (if too small)
    :return: numpy array in desired shape
    """

    if len(img_array.shape) != len(shape):
        raise ValueError(
            f"Number of dimensions have to match ({img_array.shape} != {shape})"
        )

    img_array = img_array[: shape[0], : shape[1], : shape[2]]
    img_pads = [(0, max(0, n - img_array.shape[i])) for i, n in enumerate(shape)]
    img_array = np.pad(
        img_array, img_pads, mode="constant", constant_values=padding_value
    )

    return img_array


def generate_window_polygon(datasource: DatasetReader, window: Window) -> Polygon:
    """Turns the spatial bounds of a given window to a shapely.Polygon object in a given
    dataset's CRS.

    :param datasource: a rasterio DatasetReader object that provides the affine
        transformation
    :param window: bounds to represent as Polygon
    :return: shapely Polygon representing the spatial bounds of a given window in a
        given CRS
    """

    window_transform = datasource.window_transform(window)
    window_bounds = array_bounds(window.height, window.width, window_transform)
    window_poly = box(*window_bounds)
    return window_poly


def generate_window_offsets(window: Window, schema: WindowSchema) -> np.ndarray:
    """Computes an array of window offsets bound by a given window.

    :param window: the bounding window (i.e. offsets will be within its bounds)
    :param schema: the parameters for the window generator
    :return: an array of window offsets within the bounds of window
    """

    col_range: np.ndarray = np.arange(
        np.max((0, window.col_off - schema.width_overlap)),
        window.width + window.col_off - schema.width_overlap,
        schema.width_step,
    )
    row_range: np.ndarray = np.arange(
        np.max((0, window.row_off - schema.height_overlap)),
        window.height + window.row_off - schema.height_overlap,
        schema.height_step,
    )

    window_offsets = np.array(np.meshgrid(col_range, row_range))
    window_offsets = window_offsets.T.reshape(-1, 2)

    return window_offsets


def window_factory(
    parent_window: Window, schema: WindowSchema, boundless: bool = True
) -> Generator[Window, None, None]:
    """Generator that produces rasterio.Window objects in predetermined steps, within
    the given Window.

    :param parent_window: the window that provides the bounds for all child_window
        objects
    :param schema: the parameters that determine the window steps
    :param boundless: whether the child_window should be clipped by the parent_window or
        not
    :yield: a rasterio.Window used for windowed reading/writing
    """

    window_offsets = generate_window_offsets(window=parent_window, schema=schema)

    for col_off, row_off in window_offsets:
        child_window = Window(
            col_off=int(col_off),
            row_off=int(row_off),
            width=int(schema.width_window),
            height=int(schema.height_window),
        )
        if not boundless:
            child_window = child_window.intersection(parent_window)

        yield child_window


def estimate_average_bounds(
    gdf: gpd.GeoDataFrame, quantile: float = 0.9
) -> Tuple[float, float]:
    """Estimates the average size of all features in a GeoDataFrame.

    :param gdf: GeoDataFrame that contains all features (i.e. shapely.Geometry objects)
    :param quantile: what quantile will represent the feature population
    :return: a tuple of floats representing average width and height
    """

    label_bounds = gdf.bounds
    label_widths = label_bounds.apply(lambda row: row["maxx"] - row["minx"], axis=1)
    label_heights = label_bounds.apply(lambda row: row["maxy"] - row["miny"], axis=1)
    average_width = np.nanquantile(label_widths, quantile).astype(float)
    average_height = np.nanquantile(label_heights, quantile).astype(float)

    return average_width, average_height


def estimate_schema(
    gdf: gpd.GeoDataFrame,
    src: DatasetReader,
    quantile: float = 0.9,
    window_bounds: List[Tuple[int, int]] = [(256, 256), (512, 512)],
) -> WindowSchema:
    """Attempts to find a schema that is able to represent the average GeoDataFrame
    feature (i.e. sufficient overlap) but within the bounds given by window_bounds.

    :param gdf: GeoDataFrame that contains features that determine the degree of overlap
    :param src: The rasterio DataSource associated with the resulting schema (i.e.
        bounds and pixelsizes)
    :param quantile: what quantile will represent the feature population
    :param window_bounds: a list of possible limits for the window generators
    :return: (if found) a viable WindowSchema with sufficient overlap within the
        window_bounds
    """

    # estimating the required overlap between windows for labels to be represented fully
    average_width, average_height = estimate_average_bounds(gdf=gdf, quantile=quantile)
    width_overlap = int(np.ceil(average_width / src.res[0]))
    height_overlap = int(np.ceil(average_height / src.res[1]))

    schema = None
    last_exception = None
    for i in range(len(window_bounds)):
        width, height = window_bounds[i]
        try:
            schema = WindowSchema(
                width_window=width,
                width_overlap=width_overlap,
                width_step=None,
                height_window=height,
                height_overlap=height_overlap,
                height_step=None,
            )
            break  # Break the loop as soon as a valid schema is found

        except ValueError as value_error:
            last_exception = value_error
            continue

    if schema is None:
        raise ValueError(
            "No WindowSchema objects could be created from the given "
            f"window_bounds {window_bounds}, raising last Exception.."
        ) from last_exception

    return schema


def assert_valid_categories(
    categories: np.ndarray, max_dtype: str = "<U50"
) -> np.ndarray:
    """Checks if all elements in categories array can be represented by strings of a
    certain length (defaults to <U50)

    :param categories: numpy array containing category values
    :param max_dtype: numpy str dtype with char size
    """

    # checking if categories is castable to str (a prerequisite for class_names)
    if not isinstance(categories, np.ndarray):
        raise ValueError("Categories needs to be of type np.ndarray")

    try:
        str_categories = categories.astype(str)
    except Exception as e:
        raise ValueError("Category values need to be castable to str") from e

    # checking if categories can be castable to str of a certain length (e.g. <U50)
    if not np.can_cast(str_categories, max_dtype):
        raise ValueError(f"Category values (str) have to fit in {max_dtype}")

    return str_categories.astype(max_dtype)


def valid_category_id(categories: np.ndarray) -> bool:
        """
        Check whether values from a given array can be cast to category_id

        :param categories: numpy array containing category values
        :return: boolean indicating castable category_id column
        """
        try:
            # attemping to cast to float (from arbitrary type)
            categories = categories.astype(float)
            # checking if castable to int without precision loss
            valid = np.all(categories.astype(int) == categories)
        except ValueError as ve:
            valid = False
        return valid
    
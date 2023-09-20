import cv2
from pycocotools import mask as cocomask
import pathlib
from typing import List, Tuple
import geopandas as gpd
import numpy as np
import rasterio
from typing import Optional
from rasterio.io import DatasetReader
from rasterio.mask import mask as riomask
from shapely.geometry import MultiPolygon
from tqdm import tqdm
from geococo.coco_models import Annotation, CocoDataset, Image
from geococo.utils import (
    estimate_schema,
    generate_window_offsets,
    window_factory,
    generate_window_polygon,
    reshape_image,
    window_intersect,
    mask_label,
    validate_labels,
    update_labels,
)


def labels_to_dataset(
    dataset: CocoDataset,
    images_dir: pathlib.Path,
    src: DatasetReader,
    window_bounds: List[Tuple[int, int]],
    labels: gpd.GeoDataFrame,
    category_id_col: Optional[str] = "category_id",
    category_name_col: Optional[str] = None,
    supercategory_col: Optional[str] = None,
) -> CocoDataset:
    """Move across a given geotiff, converting all intersecting labels to COCO
    annotations and appending them to a COCODataset model. This is done through
    rasterio.Window objects, the bounds of which you can set with window_bounds (also
    determines the size of the output images associated with the Annotation instances).
    The degree of overlap between these windows is determined by the dimensions of the
    given labels to maximize representation in the resulting dataset.

    The "iscrowd" attribute (see https://cocodataset.org/#format-data) is determined by
    whether    the respective labels are Polygon or MultiPolygon instances. The
    "category_id" attribute,    which represents class or category identifiers, is
    expected to be present in the given labels    GeoDataFrame under the same name.

    :param dataset: CocoDataset model to append images and annotations to
    :param images_dir: output directory for all label images
    :param src: open rasterio reader for input raster
    :param labels: GeoDataFrame containing labels and class_info ('category_id')
    :param window_bounds: a list of window_bounds to attempt to use ()
    :param category_id_col: Column containing category_id values
    :param category_name_col: Column containing category_name values
    :param supercategory_col: Column containing supercategory values
    :return: The COCO dataset with appended Images and Annotations
    """

    # checks presence, types and values of all required attributes
    labels = validate_labels(
        labels=labels,
        category_id_col=category_id_col,
        category_name_col=category_name_col,
        supercategory_col=supercategory_col,
    )

    # adding new Category instances from labels (if any)
    dataset.add_categories(
        category_ids=labels.get(category_id_col),
        category_names=labels.get(category_name_col),
        super_names=labels.get(supercategory_col),
    )

    # updating labels with validated COCO keys (i.e. 'name', 'id', 'supercategory')
    labels = update_labels(
        labels=labels,
        categories=dataset.categories,
        category_id_col=category_id_col,
        category_name_col=category_name_col,
    )

    # Setting nodata and estimating window configuration
    parent_window = window_intersect(input_raster=src, input_vector=labels)
    nodata_value = src.nodata if src.nodata else 0
    coco_profile = src.profile.copy()
    coco_profile.update({"dtype": np.uint8, "nodata": nodata_value, "driver": "JPEG"})
    schema = estimate_schema(gdf=labels, src=src, window_bounds=window_bounds)
    n_windows = generate_window_offsets(window=parent_window, schema=schema).shape[0]

    # sets dataset.next_source_id and bump either minor or patch version
    dataset.add_source(source_path=pathlib.Path(src.name))

    # bumps major version if images_dir has been used in this dataset before
    dataset.verify_new_output_dir(images_dir=images_dir)

    for child_window in tqdm(
        window_factory(parent_window=parent_window, schema=schema), total=n_windows
    ):
        # changing window to Shapely.geometry, used to check for intersecting labels
        window_geom = generate_window_polygon(datasource=src, window=child_window)
        intersect_mask = labels.intersects(window_geom)
        if not intersect_mask.any():
            continue

        # all_touched is needed because of np.ceil in WindowSource
        window_labels = labels[intersect_mask]
        window_image, window_transform = riomask(
            dataset=src, shapes=[window_geom], all_touched=True, crop=True
        )

        # padding is needed because rasterio clips any mask to the extent of the source
        window_shape = (src.count, child_window.width, child_window.height)
        if window_image.shape != window_shape:
            window_image = reshape_image(
                img_array=window_image, shape=window_shape, padding_value=nodata_value
            )

        # normalizing values to uint8 range (i.e COCO dtype)
        if window_image.dtype != np.uint8:
            window_image = cv2.normalize(
                window_image,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,  # type: ignore
            )

        # updating rasterio profile with window-specific values
        coco_profile.update(
            {
                "width": window_image.shape[1],
                "height": window_image.shape[2],
                "transform": window_transform,
            }
        )

        # saving window_image to disk (if it doesn't exist already)
        window_image_path = (
            images_dir / f"{dataset.next_source_id}_{child_window.col_off}_"
            f"{child_window.row_off}_{child_window.width}_{child_window.height}.jpg"
        )
        if not window_image_path.exists():
            with rasterio.open(window_image_path, "w", **coco_profile) as dst:
                dst.write(window_image)

        # Instancing and adding Image model to dataset (also bumps next_image_id)
        image_instance = Image(
            id=dataset.next_image_id,
            width=window_image.shape[1],
            height=window_image.shape[2],
            file_name=window_image_path,
            source_id=dataset.next_source_id,
        )

        # Iteratively add Annotation models to dataset (also bumps next_annotation_id)
        with rasterio.open(window_image_path) as windowed_src:
            for _, window_label in window_labels.sort_values("id").iterrows():
                label_mask = mask_label(
                    input_raster=windowed_src, label=window_label.geometry
                )
                if not label_mask.any():
                    continue

                rle = cocomask.encode(np.asfortranarray(label_mask))
                bounding_box = cv2.boundingRect(label_mask.astype(np.uint8))
                area = np.sum(label_mask)
                iscrowd = 1 if isinstance(window_label.geometry, MultiPolygon) else 0

                annotation_instance = Annotation(
                    id=dataset.next_annotation_id,
                    image_id=dataset.next_image_id,
                    category_id=window_label["id"],
                    segmentation=rle,  # type: ignore
                    area=area,
                    bbox=bounding_box,
                    iscrowd=iscrowd,
                )

                dataset.add_annotation(annotation=annotation_instance)
        dataset.add_image(image=image_instance)
    return dataset

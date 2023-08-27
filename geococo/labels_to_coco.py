import pathlib
import tempfile
import uuid
from datetime import datetime
from typing import List, Tuple
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.mask import mask as riomask
from shapely.geometry import box
from tqdm import tqdm
from geococo.models import Annotation, Category, CocoDataset, Image, Info
from geococo.utils import estimate_window_source, generate_window_offsets, window_factory, generate_window_polygon, reshape_image, process_label_mask


def populate_coco_dataset(dataset: CocoDataset, images_dir: pathlib.Path, src: DatasetReader, labels: gpd.GeoDataFrame, window_bounds:  List[Tuple[int, int]]) -> CocoDataset:    
    
    # Setting nodata and estimating window configuration
    nodata_value = src.nodata if src.nodata else 0    
    coco_profile = src.profile.copy()
    coco_profile.update({'dtype': np.uint8, 'nodata': nodata_value, "driver": "JPEG"})
    window_source = estimate_window_source(gdf=labels, src=src, window_bounds = window_bounds)
    parent_window = src.window(*src.bounds)
    n_windows = generate_window_offsets(window = parent_window, window_source = window_source).shape[0]

    for child_window in tqdm(window_factory(parent_window=parent_window, window_source=window_source), total=n_windows):

        # changing window to Shapely.geometry, used to check for intersecting labels
        window_geom = generate_window_polygon(datasource=src, window = child_window)
        intersect_mask = labels.intersects(window_geom)
        if not intersect_mask.any():
            continue
        
        # all_touched is needed because of np.ceil in WindowSource
        window_labels = labels[intersect_mask]
        window_image, window_transform = riomask(dataset=src, shapes=[window_geom], all_touched=True, crop=True)
        
        # padding is needed because rasterio clips any mask to the extent of the given datasource        
        window_shape = (src.count, child_window.width, child_window.height)
        if window_image.shape != window_shape:
            window_image = reshape_image(img_array=window_image, shape=window_shape, padding_value=nodata_value)

        # normalizing values to uint8 range (i.e COCO dtype) 
        if window_image.dtype != np.uint8:
            window_image = cv2.normalize(window_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # updating rasterio profile with window-specific values
        coco_profile.update({'width': window_image.shape[1], 'height': window_image.shape[2], 'transform': window_transform})
        
        # saving window_image to disk (if it doesn't exist already)
        window_image_path = images_dir / f"{child_window.col_off}_{child_window.row_off}_{child_window.width}_{child_window.height}.jpg"
        if not window_image_path.exists():
            with rasterio.open(window_image_path, 'w', **coco_profile) as dst:
                dst.write(window_image)
        
        # Instancing and adding Image model to dataset (also bumps next_image_id) 
        image_instance = Image(
            id=dataset.next_image_id,
            width=window_image.shape[1],
            height=window_image.shape[2],
            file_name = window_image_path
            )
        dataset.add_image(image=image_instance)
        
        # Iteratively add Annotation models to dataset (also bumps next_annotation_id)
        with rasterio.open(window_image_path) as windowed_src:
            for _, window_label in window_labels.sort_values('category_id').iterrows():
                label_mask, _ = riomask(dataset=windowed_src, shapes = [window_label.geometry], all_touched=True, filled=False)
                rle, bounding_box, area = process_label_mask(label_mask=label_mask)
                
                annotation_instance = Annotation(
                    id = dataset.next_annotation_id,
                    image_id = dataset.next_image_id,
                    category_id=window_label['category_id'],
                    segmentation=rle,
                    area=area,
                    bbox = bounding_box,
                    iscrowd=0
                    )
                dataset.add_annotation(annotation=annotation_instance)
                
    return dataset
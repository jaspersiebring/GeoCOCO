import pathlib
import tempfile
import uuid
from datetime import datetime
from typing import List, Tuple
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pycocotools import mask as cocomask
from rasterio.io import DatasetReader
from rasterio.mask import mask as riomask
from rasterio.transform import array_bounds
from shapely.geometry import box
from tqdm import tqdm
from geococo.models import Annotation, Category, CocoDataset, Image, Info
from geococo.utils import estimate_window_source, generate_window_offsets, window_factory


def populate_coco_dataset(dataset: CocoDataset, images_dir: pathlib.Path, src: DatasetReader, labels: gpd.GeoDataFrame, window_bounds:  List[Tuple[int, int]], normalize: bool = True) -> CocoDataset:    

    # Setting nodata and estimating window configuration
    dst_profile = src.profile.copy()
    nodata_value = src.nodata if src.nodata else 0    
    window_source = estimate_window_source(gdf=labels, src=src, window_bounds = window_bounds)
    parent_window = src.window(*src.bounds)
    n_windows = generate_window_offsets(window = parent_window, window_source = window_source).shape[0]

    geoms = [] 
    for child_window in tqdm(window_factory(parent_window=parent_window, window_source=window_source), total=n_windows):

        # changing window to Shapely.geometry, used to check for intersecting labels
        window_transform = src.window_transform(child_window)
        window_bounds = array_bounds(child_window.height, child_window.width, window_transform)
        window_geom = box(*window_bounds)
        intersect_mask = labels.intersects(window_geom)
        if not intersect_mask.any():
            continue
        
        # all_touched is needed because of np.ceil in WindowSource
        intersecting_labels = labels[intersect_mask]
        window_image, _ = riomask(
            dataset=src,
            shapes=[window_geom],
            all_touched=True,
            filled=True,
            crop=True,
            nodata=nodata_value
        )
        
        # padding is needed because rasterio clips any mask to the extent of the given datasource
        if window_image.shape != (src.count, child_window.width, child_window.height):
            window_image = window_image[:src.count, :child_window.width, :child_window.height]
            pad_dims = [(0, max(0, n - window_image.shape[i])) for i, n in enumerate([src.count, child_window.width, child_window.height])]
            window_image = np.pad(window_image, pad_dims, mode='constant', constant_values=nodata_value)
        
        # normalizing values to uint8 range (i.e COCO dtype) 
        if window_image.dtype != np.uint8 and normalize:
            window_image = cv2.normalize(window_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # updating profile with changed values      
        dst_profile.update(
            {
                'dtype': np.uint8,
                'width': window_image.shape[1],
                'height': window_image.shape[2],
                'nodata': nodata_value,
                'transform': window_transform,
                "driver": "JPEG"
            }
        )
        
        # saving window_image to disk (if it doesn't exist already)
        window_image_path = images_dir / f"{child_window.col_off}_{child_window.row_off}_{child_window.width}_{child_window.height}.jpg"
        if not window_image_path.exists():
            with rasterio.open(window_image_path, 'w', **dst_profile) as dst:
                dst.write(window_image)
        
        # Instancing and adding Image model to dataset (also bumps next_image_id) 
        image_instance = Image(
            id=dataset.next_image_id,
            width=window_image.shape[1],
            height=window_image.shape[2],
            file_name = window_image_path)
        dataset.add_image(image=image_instance)
        
        # Iteratively add Annotation models to dataset (also bumps next_annotation_id)
        with rasterio.open(window_image_path) as windowed_src:
            for _, intersecting_label in intersecting_labels.sort_values('category_id').iterrows():
                intersected_image, _ = riomask(
                    dataset=windowed_src,
                    shapes = [intersecting_label.geometry], 
                    all_touched=True,
                    filled=False
                )
                
                intersected_mask = np.all(intersected_image.mask, axis=0)
                intersected_mask = np.invert(intersected_mask)
                rle = cocomask.encode(np.asfortranarray(intersected_mask))
                bounding_box = cv2.boundingRect(intersected_mask.astype("uint8"))
                area = np.sum(intersected_mask)
                category_id = intersecting_label['category_id']
                iscrowd = 0 # TODO determined by multipolygon

                annotation_instance = Annotation(
                    id = dataset.next_annotation_id,
                    image_id = dataset.next_image_id,
                    category_id=category_id,
                    segmentation=rle,
                    area=area,
                    bbox = bounding_box,
                    iscrowd=iscrowd)
                dataset.add_annotation(annotation=annotation_instance)
                
        geoms.append(window_geom)

    return dataset, geoms
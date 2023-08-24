import pathlib
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pycocotools import mask as cocomask
from rasterio.io import DatasetReader
from rasterio.mask import mask as riomask
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm
from geococo.models import Annotation, Category, CocoDataset, Image, Info, WindowSource
from geococo.utils import estimate_window_source, generate_window_offsets, window_factory, dump_dataset, load_dataset


def populate_coco_dataset(dataset: CocoDataset, images_dir: pathlib.Path, src: DatasetReader, labels: gpd.GeoDataFrame, window_bounds:  List[Tuple[int, int]]) -> CocoDataset:
    image_ids = [image.id for image in dataset.images]
    annotation_ids = [annotation.id for annotation in dataset.annotations]
    next_image_id = 1 if not image_ids  else np.max(image_ids)
    next_annotation_id = 1 if not annotation_ids else np.max(annotation_ids)
    
    dst_profile = src.profile.copy()
    nodata_value = src.nodata if src.nodata else 0
    
    window_source = estimate_window_source(gdf=labels, src=src, window_bounds = window_bounds)
    parent_window = src.window(*src.bounds)
    n_windows = generate_window_offsets(window = parent_window, window_source = window_source).shape[0]

    geoms = []
    for child_window in tqdm(window_factory(parent_window=parent_window, window_source=window_source), total=n_windows):
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
                
        # updating profile with changed values
        dst_profile.update(
            {
                'dtype': window_image.dtype,
                'width': window_image.shape[1],
                'height': window_image.shape[2],
                'nodata': nodata_value,
                'transform': window_transform,
                "driver": "JPEG"
            }
        )

        window_image_path = images_dir / f"{child_window.col_off}_{child_window.row_off}_{child_window.width}_{child_window.height}.jpg"
        
        if not window_image_path.exists():
            with rasterio.open(window_image_path, 'w', **dst_profile) as dst:
                dst.write(window_image)
    
        image_instance = Image(
            id=next_image_id,
            width=window_image.shape[1],
            height=window_image.shape[2],
            file_name = window_image_path)
        
        dataset.images.append(image_instance)
        
        # adding Annotation objects
        with rasterio.open(window_image_path) as windowed_src:
            for _, intersecting_label in intersecting_labels.sort_values('category_id').iterrows():
                intersected_image, intersected_transform = riomask(
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
                    id=next_annotation_id,
                    image_id = next_image_id,
                    category_id=category_id,
                    segmentation=rle,
                    area=area,
                    bbox = bounding_box,
                    iscrowd=iscrowd)
                
                annotation_instance
                dataset.annotations.append(annotation_instance)
                next_annotation_id += 1

        next_image_id += 1
        geoms.append(window_geom)

    return dataset, geoms


    
if __name__ == "__main__":
    image_path = pathlib.Path("D:/repos/GeoCOCO/data/image.tif")
    label_path = pathlib.Path("D:/repos/GeoCOCO/data/labels.shp")
    
    labels: gpd.GeoDataFrame = gpd.read_file(label_path)
    labels['category_id'] = np.random.randint(1, 10, labels.shape[0]) #faking categories
    src: DatasetReader = rasterio.open(image_path)
    
    window_bounds = [(256, 256), (512, 512)]
    raster_bounds = gpd.GeoSeries(box(*src.bounds), crs=src.crs)
    label_bounds = gpd.GeoSeries(box(*labels.total_bounds), crs=labels.crs)
        
    if not src.crs.to_string() == labels.crs.to_string():
        raise ValueError("Projection of given spatial objects don't match, exiting..")

    #check for overlapping bounds (requirement)
    if not all(raster_bounds.intersects(label_bounds)):
        raise ValueError("There's no overlap between input objects, exiting..")


    output_dir_str = tempfile.TemporaryDirectory(prefix=f"output_{uuid.uuid4()}")
    output_dir = pathlib.Path(output_dir_str.name)

    print(output_dir)

    # Instancing new dataset
    info = Info(version = "0.0.1", date_created=datetime.now())
    dataset = CocoDataset(
        info=info,
        images = [],
        annotations = [],
        categories=[]
        )

    
    dataset, geoms = populate_coco_dataset(
        dataset = dataset, 
        images_dir = output_dir,
        src = src,
        labels = labels,
        window_bounds = window_bounds
        )
    
    
    gpd_path = output_dir / "geoms.shp"
    gpd.GeoDataFrame(geometry = geoms, crs = src.crs).to_file(gpd_path)
    
        
    json_path = output_dir / "first_dataset.json"
    dump_dataset(dataset=dataset, json_path=json_path)
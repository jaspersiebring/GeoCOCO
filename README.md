![geococo logo](https://github.com/jaspersiebring/GeoCOCO/assets/25051531/b2a2db16-1400-4c43-b044-a924a378ef84)
---
[![PyPI](https://img.shields.io/pypi/v/geococo)](https://pypi.org/project/geococo/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/geococo)
[![Build passing](https://github.com/jaspersiebring/GeoCOCO/actions/workflows/main.yml/badge.svg)](https://github.com/jaspersiebring/GeoCOCO/actions/workflows/main.yml)

Easily transform your GIS annotations into [Microsoft's Common Objects In Context (COCO)](https://cocodataset.org/#format-data) datasets with GeoCOCO. This tool allows users to leverage the advanced digitizing solutions of modern GIS software for the annotations of image objects in geographic imagery.
 
Built with `Pydantic` and `pycocotools`, it features a complete implementation of the COCO standard for object detection with out-of-the-box support for JSON-encoding and RLE compression. The resulting datasets are versioned, easily extendable with new annotations and fully compatible with other data applications that accept the COCO format.

# Key features
- User-friendly: GeoCOCO is designed for ease of use, requiring minimal configuration and domain knowledge
- Version Control: Datasets created with GeoCOCO are versioned and designed for expansion with future annotations
- Command-line Tool: Use GeoCOCO from your terminal for quick conversions
- Python Module: Integrate GeoCOCO in your own data applications with the `geococo` package
- Representation: GeoCOCO maximizes label representation through an adaptive moving window approach
- COCO Standard: Output datasets are fully compatible with other COCO-accepting applications
- Compact File Size: JSON-encoding and RLE compression are employed to ensure compact file sizes

# Installation
Installing from the Python Package Index (PyPI):
````
# Install from PYPI with Python's package installer (pip)
pip install geococo
````

## Example of usage

After installing `geococo`, there are a number of ways you can interact with its API.

#### Command line interface
 The easiest way to use `geococo` is to simply call it from your preferred terminal. You can use the tool entirely from your terminal by providing paths to your input data and the desired output image sizes like this.

  ````
# Example with local data and non-existent JSON file
geococo image.tif labels.shp coco_folder dataset.json 512 512

Creating new dataset..
Dataset version: 0.1.0
Dataset description: Test dataset
Dataset contributor: User
Dataset date: 2023-09-05 18:12:31.435591
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:04<00:00, 50.36it/s]
````
For more information on the different options, call `geococo` with `--help`
````
geococo --help
Usage: cli.py [OPTIONS] IMAGE_PATH LABELS_PATH JSON_PATH OUTPUT_DIR WIDTH HEIGHT

  Transform your GIS annotations into a COCO dataset.

  This method generates a COCO dataset by moving across the given image
  (image_path) with a moving window (image_size), constantly checking for
  intersecting annotations (labels_path)  that represent image objects in said
  image (e.g. buildings in satellite imagery; denoted  by category_attribute).
  Each valid intersection will add n Annotations entries to the dataset
  (json_path) and save a subset of the input image that contained these entries
  (output_dir).

  The output data size depends on your input labels, as the moving window
  adjusts its step size  to accommodate the average annotation size, optimizing
  dataset representation and minimizing  tool configuration.

Arguments:
  IMAGE_PATH                 Path to the geospatial image containing image
                             objects (e.g. buildings in satellite imagery)
                             [required]
  LABELS_PATH                Path to the annotations representing these image
                             objects (='category_id')  [required]
  JSON_PATH                  Path to the json file that will store the COCO
                             dataset (will be appended to if already exists)
                             [required]
  OUTPUT_DIR                 Path to the output directory for image subsets
                             [required]
  WIDTH                      Width of the output images  [required]
  HEIGHT                     Height of the output images  [required]

Options:
  --category-attribute TEXT  Column that contains category_id values per
                             annotation feature  [default: category_id]
  --help                     Show this message and exit.

````

#### Python module
This is recommended for most developers as it gives you more granular control over the various steps. It does assume a basic understanding of the `geopandas` and `rasterio` packages.

````
import geopandas as gpd
import rasterio
from datetime import datetime
from geococo import create_dataset, load_dataset, save_dataset, labels_to_dataset

# Replace this with your preferred output paths
data_path = pathlib.Path("path/to/your/coco/output/images")
json_path = pathlib.Path("path/to/your/coco/json/file")

# Dimensions of the moving window and output images
width, height = 512, 512

# Creating dataset instance from scratch
description = "My First Dataset"
contributor = "User'
date_created = datetime.now()

dataset = create_dataset(
  version = version, 
  description = description, 
  contributor = contributor, 
  date_created = date_created
)

# You can also load existing COCO datasets
# dataset = load_dataset(json_path=json_path)

# Loading GIS data with rasterio and geopandas
labels = gpd.read_file(labels_path)
raster_source = rasterio.open(image_path)

# Moving across raster_source and appending all intersecting annotations
dataset = labels_to_dataset(
    dataset = dataset, 
    images_dir = output_dir,
    src = raster_source,
    labels = labels,
    window_bounds = [(width, height)]
    )

# Encode CocoDataset instance as JSON and save to json_path
save_dataset(dataset=dataset, json_path=json_path)
````

# Visualization with FiftyOne
Like the official COCO project, the open source tool [FiftyOne](https://docs.voxel51.com/) can be used to visualize and evaluate your datasets. This does require the `fiftyone` and `pycocotools` packages (the former of which is not installed by `geococo` so you would need to install this separately, see https://docs.voxel51.com/getting_started/install.html for instructions). After installing `fiftyone`, you can run the following to inspect your data in your browser.

````
# requires pycocotools and fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
import pathlib

data_path = pathlib.Path("path/to/your/coco/output/images")
json_path = pathlib.Path("path/to/your/coco/json/file")

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=json_path,
    include_id=True,
)

# Launch the app
session = fo.launch_app(coco_dataset, port=5151)
````

<p float="left">
  <img src="https://github.com/jaspersiebring/GeoCOCO/assets/25051531/f8ab55da-b3cd-4beb-b082-7946e712ea5c" width="45%" height = 250/>
  <img src="https://github.com/jaspersiebring/GeoCOCO/assets/25051531/9a796a54-ffc2-49c3-95bc-59e5c0dd1d7c" width="45%" height = 250 />
</p>


# Planned features
- [QGIS plugin](https://github.com/jaspersiebring/geococo-qgis-plugin).
- Data visualization with `pycocotool`'s plotting functionality

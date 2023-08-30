![geococo logo](https://github.com/jaspersiebring/GeoCOCO/assets/25051531/a5749794-7314-4a17-ae71-3d8015a5933c)

[![PyPI](https://img.shields.io/pypi/v/libretro-finder)](https://pypi.org/project/libretro-finder/)
[![Downloads](https://static.pepy.tech/badge/libretro-finder)](https://pepy.tech/project/libretro-finder)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/libretro-finder)
![PyPI - License](https://img.shields.io/pypi/l/libretro-finder)

Tool for converting GIS annotations to Microsoft's Common Objects In Context (COCO) datasets (see https://cocodataset.org/#format-data). Inspired by [labelme](https://github.com/wkentaro/labelme), it sets out to simplify the process of preparing deep learning datasets through the mature digitizing solutions of modern GIS software like QGIS and ArcGIS Pro. This is done in part by taking advantage of the spatial properties of GIS annotations to maximize their representation in the resulting COCO dataset and to minimize configuration of the tool itself. Just specify the desired image size and go.

Built with `Pydantic` and `pycocotools`, it features a 1:1 representation of the COCO standard with out-of-the-box support for JSON-encoding and RLE compression. The resulting datasets are versionable and designed to be extended by subsequent runs with new data. It also accepts any GIS file format as input, courtesy of the `geopandas` and `rasterio` packages (e.g. geotif, geopackage, shapefile).

Lastly, it can be run directly from your terminal as a standalone command line tool or be integrated into your image processing pipeline via the [`geococo` module](https://pypi.org/project/libretro-finder/).


# Features
- User-friendly, minimal configuration needed
- Output datasets are versioned and extendable
- Use any geotiff as input image (windowed reading)
- Maximum label representation due to adaptive moving window
- Small footprint due to RLE compression
- Usable as Python module or as QGIS plugin
- Command line interface


# Installation
Installing from the Python Package Index (PyPI):
````
# Install from PYPI with Python's package installer (pip)
pip install geococo

# [Optional] Install as isolated application through pipx (https://pypa.github.io/pipx/)
pipx install geococo
````


# Using `geococo` in Python
After installing `geococo`, the following methods can be used to create and save COCO datasets from your own GIS annotations.
````
# Importing main interface
import pathlib
from geococo import load_model, create_model, append_model

# Replace this with your own input data
raster_path = pathlib.Path("path/to/your/input/raster")
label_path = pathlib.Path("path/to/your/input/labels")

# Replace this with your preferred output paths
data_path = pathlib.Path("path/to/your/coco/output/images")
json_path = pathlib.Path("path/to/your/coco/json/file")

# Instancing existing model
dataset = load_model()

# Instancing new model
dataset = create_model()

# Processing and
dataset = append_model(
    dataset=dataset,
    raster_path = raster_path,
    label_path = label_path,
    bounds = [512, 512]
    )
````


# Visualization with FiftyOne
Just like the official COCO project, the open source tool [FiftyOne](https://docs.voxel51.com/) can be used to visualize and evaluate your `geococo` datasets. This does require the `fiftyone` and `pycocotools` packages to be installed (the former of which is not installed by `geococo` so needs to be installed separately). After installing `fiftyone` (see https://docs.voxel51.com/getting_started/install.html), you can run the following to inspect your data in your browser.

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
  <img src="https://github.com/jaspersiebring/GeoCOCO/assets/25051531/3fd31779-7c9b-4524-9722-44b4f693a023" width="45%" height = 250/>
  <img src="https://github.com/jaspersiebring/GeoCOCO/assets/25051531/71ed2d59-d6b1-426a-9c7a-1260a50e4e40" width="45%" height = 250 />
</p>



# Planned features
- [QGIS plugin](https://github.com/jaspersiebring/geococo-qgis-plugin).
- Data visualization with `pycocotool`'s plotting functionality

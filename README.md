![geococo logo](https://github.com/jaspersiebring/GeoCOCO/assets/25051531/6a7b9912-4276-49dc-a09b-fdbea11f6294)
---
[![PyPI](https://img.shields.io/pypi/v/libretro-finder)](https://pypi.org/project/libretro-finder/)
[![Downloads](https://static.pepy.tech/badge/libretro-finder)](https://pepy.tech/project/libretro-finder)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/libretro-finder)
![PyPI - License](https://img.shields.io/pypi/l/libretro-finder)

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

# [Optional] Install as isolated application through pipx (https://pypa.github.io/pipx/)
pipx install geococo
````

## Example of usage

#### Python module
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

#### Command line interface

If installed with pip, GeoCOCO can be called directly from your preferred terminal by running `geococo`. You can use the tool entirely from your terminal by providing paths to your input data and the desired image size like this.

````
some_user@some_machine:~ geococo --input_raster /your/input/raster --input_labels /your/input/labels --size 256,256 --output_dir /your/output/images --output_json /your/output/json
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
  <img src="https://github.com/jaspersiebring/GeoCOCO/assets/25051531/3fd31779-7c9b-4524-9722-44b4f693a023" width="45%" height = 250/>
  <img src="https://github.com/jaspersiebring/GeoCOCO/assets/25051531/71ed2d59-d6b1-426a-9c7a-1260a50e4e40" width="45%" height = 250 />
</p>


# Planned features
- [QGIS plugin](https://github.com/jaspersiebring/geococo-qgis-plugin).
- Data visualization with `pycocotool`'s plotting functionality

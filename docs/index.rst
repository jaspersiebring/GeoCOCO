.. geococo documentation master file, created by
   sphinx-quickstart on Mon Sep 25 01:26:39 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to geococo's documentation!
===================================
.. image:: https://github.com/jaspersiebring/GeoCOCO/assets/25051531/b2a2db16-1400-4c43-b044-a924a378ef84
   :alt: GeoCOCO logo

Easily transform your GIS annotations into `Microsoft's Common Objects In Context (COCO) <https://cocodataset.org/#format-data>`_ datasets with GeoCOCO. This tool allows users to leverage the advanced digitizing solutions of modern GIS software for the annotations of image objects in geographic imagery.
 
Built with `Pydantic` and `pycocotools`, it features a complete implementation of the COCO standard for object detection with out-of-the-box support for JSON-encoding and RLE compression. The resulting datasets are versioned, easily extendable with new annotations and fully compatible with other data applications that accept the COCO format.

.. code:: python

   import pathlib
   import geopandas as gpd
   import rasterio
   from geococo import create_dataset, load_dataset, save_dataset, append_dataset

   # Replace this with your preferred output paths
   data_path = pathlib.Path("path/to/your/coco/output/images")
   json_path = pathlib.Path("path/to/your/coco/json/file")

   # Dimensions of the moving window and output images
   width, height = 512, 512

   # Starting a new CocoDataset
   description = "My First Dataset"
   contributor = "User"

   # version and date_created are automatically set
   dataset = create_dataset(description=description, contributor=contributor)
   # You can also load existing COCO datasets
   # dataset = load_dataset(json_path=json_path)

   # Loading GIS data with rasterio and geopandas
   labels = gpd.read_file(some_labels_path)
   raster_source = rasterio.open(some_image_path)

   # (Optional) Apply any spatial or attribute queries here
   # labels = labels.loc[labels["ids"].isin([1, 2, 3])]
   # labels = labels.loc[labels.within(some_polygon)]

   # Find and save all Annotation instances
   dataset = append_dataset(
      dataset=dataset,
      images_dir=data_path,
      src=raster_source,
      labels=labels,
      window_bounds=[(width, height)],
      id_attribute=None,  # column with category_id values
      name_attribute="ids",  # column with category_name values
      super_attribute=None,  # optional column with super_category values
   )

   # Encode CocoDataset instance as JSON and save to json_path
   save_dataset(dataset=dataset, json_path=json_path)


.. toctree::
   :maxdepth: 2

   intro
   installation
   quickstart
   cli
   fiftyone
   examples
   modules
   geococo
   tests


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

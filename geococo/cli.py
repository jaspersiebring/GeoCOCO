import typer_cloup as typer
import pathlib
from datetime import datetime
import geopandas as gpd
import rasterio
from geococo.coco_processing import labels_to_dataset
from geococo.coco_manager import save_dataset, load_dataset, create_dataset


def build_coco(
    image_path: pathlib.Path,
    labels_path: pathlib.Path,
    json_path: pathlib.Path,
    output_dir: pathlib.Path,
    width: int,
    height: int,
    category_attribute: str = "category_id",
) -> None:
    """Transform your GIS annotations into a COCO dataset.

    This method generates a COCO dataset by moving across the given image (image_path)
    with a moving window (image_size), constantly checking for intersecting annotations
    (labels_path) that represent image objects in said image (e.g. buildings in
    satellite imagery; denoted by category_attribute). Each valid intersection will add
    n Annotations entries to the dataset (json_path) and save a subset of the input
    image that contained these entries (output_dir).

    The output data size depends on your input labels, as the moving window adjusts its
    step size to accommodate the average annotation size, optimizing dataset
    representation and minimizing tool configuration.

    :param image_path: Path to the geospatial image containing image objects (e.g.
        buildings in satellite imagery)
    :param labels_path: Path to the annotations representing these image objects
        (='category_id')
    :param json_path: Path to the json file that will store the COCO dataset (will be
        appended to if already exists)
    :param output_dir: Path to the output directory for image subsets
    :param width: Width of the output images
    :param height: Height of the output images
    :param category_attribute: Column that contains category_id values per annotation
        feature
    """

    if isinstance(json_path, pathlib.Path) and json_path.exists():
        # Create and populate instance of CocoDataset from json_path
        dataset = load_dataset(json_path=json_path)

    else:
        # Create instance of CocoDataset from user input
        print("Creating new dataset..")
        version = input("Dataset version: ")
        description = input("Dataset description: ")
        contributor = input("Dataset contributor: ")
        date_created = datetime.now()
        print(f"Dataset date: {date_created}")

        dataset = create_dataset(
            version=version,
            description=description,
            contributor=contributor,
            date_created=date_created,
        )

    # Loading and validating GIS data
    labels = gpd.read_file(labels_path)
    if not labels.is_valid.all():
        raise ValueError("Invalid geometry found, exiting..")
    elif category_attribute not in labels.columns:
        raise ValueError(
            f"User-specified category attribute (={category_attribute}) not found in "
            "input labels, exiting.."
        )

    with rasterio.open(image_path) as src:
        # Appending Annotation instances and clipping the image part that contains them
        dataset = labels_to_dataset(
            dataset=dataset,
            images_dir=output_dir,
            src=src,
            labels=labels,
            window_bounds=[(width, height)],
        )

    # Encode CocoDataset instance as JSON and save to json_path
    save_dataset(dataset=dataset, json_path=json_path)


if __name__ == "__main__":
    typer.run(build_coco)

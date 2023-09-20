import typer
import pathlib
from datetime import datetime
import geopandas as gpd
import rasterio
from geococo.coco_processing import labels_to_dataset
from geococo.coco_manager import save_dataset, load_dataset, create_dataset
from typing_extensions import Annotated

app = typer.Typer(add_completion=False, help="Transform your GIS annotations into COCO datasets.")

@app.command()
def new(
    json_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the json file that will store the empty COCO dataset",
            exists=False,
            file_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ]
) -> None:
    """Initialize a CocoDataset with version 0.0.0 and save it to json_path"""

    print("Creating new dataset..")
    description = input("Dataset description: ")
    contributor = input("Dataset contributor: ")

    dataset = create_dataset(
        description=description,
        contributor=contributor,
    )

    # Encode CocoDataset instance as JSON and save to json_path
    save_dataset(dataset=dataset, json_path=json_path)
    print(f"Created new CocoDataset as {json_path}")


@app.command()
def copy(
    source_path: Annotated[
        pathlib.Path, typer.Argument(help="Path to input CocoDataset")
    ],
    dest_path: Annotated[
        pathlib.Path, typer.Argument(help="Path to output CocoDataset")
    ],
    update_meta: Annotated[
        bool,
        typer.Option(
            help="Whether to prompt the user for new metadata for the copied CocoDataset"
        ),
    ] = True,
) -> None:
    """
    Copies a CocoDataset from source_path, prompts user for new metadata (optional) and saves it to dest_path
    """

    dataset = load_dataset(json_path=source_path)
    # Encode CocoDataset instance as JSON and save to json_path

    if update_meta:
        print("Updating metadata..")
        dataset.info.version = input(f"Dataset version ({dataset.info.version}): ") or dataset.info.version
        dataset.info.description = input(
            f"Dataset description ({dataset.info.description}): " or dataset.info.description
        )
        dataset.info.contributor = input(
            f"Dataset contributor ({dataset.info.contributor}): " or dataset.info.contributor
        )
        dataset.info.date_created = datetime.now()
        print(f"Dataset date: {dataset.info.date_created}")
        dataset.info.year = dataset.info.date_created.year

    save_dataset(dataset=dataset, json_path=dest_path)
    print(f"Copied CocoDataset to {dest_path}")


@app.command()
def add(
    image_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help=(
                "Path to the geospatial image containing image objects (e.g. buildings in satellite imagery)"
            ),
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    labels_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the annotations representing these image objects (='category_id')"
        ),
    ],
    json_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the json file that will store the COCO dataset (will be appended to if already exists)"
        ),
    ],
    output_dir: Annotated[
        pathlib.Path,
        typer.Argument(help="Path to the output directory for image subsets"),
    ],
    width: Annotated[int, typer.Argument(help="Width of the output images")],
    height: Annotated[int, typer.Argument(help="Height of the output images")],
    category_attribute: Annotated[
        str,
        typer.Option(
            help="Column that contains category_id values per annotation feature"
        ),
    ] = "category_id",
) -> None:
    """
    Transform and add GIS annotations to an existing COCO dataset.
    
    This method generates a COCO dataset by moving across the given image (image_path) with a
    moving window (image_size), constantly checking for intersecting annotations (labels_path)
    that represent image objects in said image (e.g. buildings in satellite imagery; denoted
    by category_attribute). Each valid intersection will add n Annotations entries to the dataset
    (json_path) and save a subset of the input image that contained these entries (output_dir).

    The output data size depends on your input labels, as the moving window adjusts its step size
    to accommodate the average annotation size, optimizing dataset representation and minimizing
    tool configuration.
    """

    # Create and populate instance of CocoDataset from json_path
    if isinstance(json_path, pathlib.Path) and json_path.exists():    
        dataset = load_dataset(json_path=json_path)
    else:
        raise ValueError("Provide existing COCO dataset through json_path")
    
    # Loading and validating GIS data
    labels = gpd.read_file(labels_path)
    if not labels.is_valid.all():
        raise ValueError("Invalid geometry found, exiting..")
    elif category_attribute not in labels.columns:
        raise ValueError(
            f"User-specified category attribute (={category_attribute}) not found "
            "in input labels"
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
    app()

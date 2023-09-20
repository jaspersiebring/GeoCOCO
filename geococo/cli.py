import typer
import pathlib
from datetime import datetime
import geopandas as gpd
import rasterio
from geococo.coco_processing import labels_to_dataset
from geococo.coco_manager import save_dataset, load_dataset, create_dataset
from typing_extensions import Annotated
from typing import Optional

app = typer.Typer(
    add_completion=False, help="Transform your GIS annotations into COCO datasets."
)


@app.command()
def new(
    json_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Output path for new CocoDataset",
            exists=False,
            file_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ]
) -> None:
    """Initialize a new CocoDataset and save it to json_path."""

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
    source_json: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to source CocoDataset",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    dest_json: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Output path for CocoDataset copy",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    update_meta: Annotated[
        bool,
        typer.Option(help="Whether to prompt the user for new metadata"),
    ] = True,
) -> None:
    """Copies a CocoDataset from source_path, prompts user for new metadata (optional)
    and saves it to dest_path."""

    # Loading CocoDataset model from json_path
    dataset = load_dataset(json_path=source_json)

    if update_meta:
        print("Updating metadata..")
        dataset.info.version = (
            input(f"Dataset version ({dataset.info.version}): ") or dataset.info.version
        )
        dataset.info.description = input(
            f"Dataset description ({dataset.info.description}): "
            or dataset.info.description
        )
        dataset.info.contributor = input(
            f"Dataset contributor ({dataset.info.contributor}): "
            or dataset.info.contributor
        )
        dataset.info.date_created = datetime.now()
        print(f"Dataset date: {dataset.info.date_created}")
        dataset.info.year = dataset.info.date_created.year

    # Encode CocoDataset instance as JSON and save to json_path
    save_dataset(dataset=dataset, json_path=dest_json)
    print(f"Copied CocoDataset to {dest_json}")


@app.command()
def add(
    image_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to geospatial image containing image objects",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    labels_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to vector file containing annotated image objects",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    json_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to json file containing the COCO dataset",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to output directory for image subsets",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    width: Annotated[int, typer.Argument(help="Width of image subsets")],
    height: Annotated[int, typer.Argument(help="Height of image subsets")],
    id_attribute: Annotated[
        Optional[str],
        typer.Option(
            help="Name of column containing category_id values "
            "(optional if --name_attribute is given)"
        ),
    ] = None,
    name_attribute: Annotated[
        Optional[str],
        typer.Option(
            help="Name of column containing category_name values "
            "(optional if --id_attribute is given)"
        ),
    ] = None,
    super_attribute: Annotated[
        Optional[str],
        typer.Option(help="Name of column containing supercategory values"),
    ] = None,
) -> None:
    """Transform and add GIS annotations to an existing COCO dataset.

    This method generates a COCO dataset by moving across the given image (image_path)
    with a moving window (image_size), constantly checking for intersecting annotations
    (labels_path) that represent image objects in said image (e.g. buildings in
    satellite imagery; denoted by (super)category name and/or id). Each valid
    intersection will add n Annotations entries to the dataset (json_path) and save a
    subset of the input image that contained these entries (output_dir).

    The output data size depends on your input labels, as the moving window adjusts its
    step size to accommodate the average annotation size, optimizing dataset
    representation and minimizing tool configuration. Each addition will also increment
    the dataset version: patch if using the same image_path, minor if using a new
    image_path, and major if using a new output_dir.
    """

    # Loading CocoDataset model from json_path
    dataset = load_dataset(json_path=json_path)

    # Loading GIS data
    labels = gpd.read_file(labels_path)
    with rasterio.open(image_path) as src:
        # Find and save all Annotation instances
        dataset = labels_to_dataset(
            dataset=dataset,
            images_dir=output_dir,
            src=src,
            labels=labels,
            window_bounds=[(width, height)],
            category_id_col=id_attribute,
            category_name_col=name_attribute,
            supercategory_col=super_attribute,
        )

    # Encode CocoDataset instance as JSON and save to json_path
    save_dataset(dataset=dataset, json_path=json_path)


if __name__ == "__main__":
    app()

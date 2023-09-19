from semver import Version
from datetime import datetime
import pathlib
from geococo.coco_models import CocoDataset, Info


def load_dataset(json_path: pathlib.Path) -> CocoDataset:
    """Dumps the contents of json_path as a string, interprets it as a CocoDataset model
    and returns it.

    :param json_path: path to the JSON file containing the json-encoded COCO dataset
    :return: An instance of CocoDataset with loaded Image- and Annotation objects from
        json_path
    """

    with open(json_path, mode="r", encoding="utf-8") as json_fp:
        json_data = json_fp.read()
    dataset = CocoDataset.parse_raw(json_data)
    return dataset


def create_dataset(
    description: str,
    contributor: str,
    version: str = str(Version(major=0)),
    date_created: datetime = datetime.now(),
) -> CocoDataset:
    """Instances and returns a new CocoDataset model with given kwargs.

    :param description: Description of your COCO dataset
    :param contributor: Main contributors of your COCO dataset, its images and its
        annotations
    :param version: Initial SemVer version (defaults to 0.0.0)
    :param date_created: Date when dataset was initially created, defaults to
        datetime.now()
    :return: An instance of CocoDataset without Image- and Annotation objects
    """

    info = Info(
        version=version,
        description=description,
        contributor=contributor,
        date_created=date_created,
        year=date_created.year,
    )
    dataset = CocoDataset(info=info)
    return dataset


def save_dataset(dataset: CocoDataset, json_path: pathlib.Path) -> None:
    """JSON-encodes an instance of CocoDataset and saves it to json_path.

    :param dataset: An instance of CocoDataset
    :param json_path: where to save the JSON-encoded CocoDataset instance to
    """

    json_data = dataset.json()
    with open(json_path, mode="w", encoding="utf-8") as dst:
        dst.write(json_data)

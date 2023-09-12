import pytest
from pydantic import ValidationError
from geococo.window_schema import WindowSchema


def test_schema_valid():
    width_window = height_window = 100
    width_overlap = height_overlap = 20

    schema = WindowSchema(
        width_window=width_window,
        width_overlap=width_overlap,
        height_window=height_window,
        height_overlap=height_overlap,
    )

    assert schema.width_step == width_window - width_overlap * 2
    assert schema.height_step == height_window - height_overlap * 2


@pytest.mark.parametrize("test_input", [(0, 20), (100, -20), (100, 50)])
def test_schema_invalid(test_input):
    window, overlap = test_input
    width_window = height_window = window
    width_overlap = height_overlap = overlap

    with pytest.raises(ValidationError):
        WindowSchema(
            width_window=width_window,
            width_overlap=width_overlap,
            height_window=height_window,
            height_overlap=height_overlap,
        )


@pytest.mark.parametrize("test_input", [(100, 20.0), (100.0, 20)])
def test_schema_float(test_input):
    window, overlap = test_input
    width_window = height_window = window
    width_overlap = height_overlap = overlap

    with pytest.raises(ValidationError):
        WindowSchema(
            width_window=width_window,
            width_overlap=width_overlap,
            height_window=height_window,
            height_overlap=height_overlap,
        )


@pytest.mark.parametrize("test_input", [("100", 20), (100, "20")])
def test_schema_str(test_input):
    window, overlap = test_input
    width_window = height_window = window
    width_overlap = height_overlap = overlap

    # TypeError is raised because 'steps' are calculated 
    # before pydantic's field validation
    with pytest.raises(TypeError):
        WindowSchema(
            width_window=width_window,
            width_overlap=width_overlap,
            height_window=height_window,
            height_overlap=height_overlap,
        )

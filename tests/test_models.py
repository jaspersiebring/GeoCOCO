from datetime import datetime
from geococo.models import WindowSource, Annotation, Category, CocoDataset, Image, License, Info
import pytest


def test_valid_overlapping_window_source():
    width_window_pixels = height_window_pixels = 100
    width_overlap_pixels = height_overlap_pixels = 20

    window_source = WindowSource(
        width_window_pixels=width_window_pixels,
        width_overlap_pixels=width_overlap_pixels,
        height_window_pixels=height_window_pixels,
        height_overlap_pixels=height_overlap_pixels
        )

    assert window_source.width_step_pixels == width_window_pixels - width_overlap_pixels * 2
    assert window_source.height_step_pixels == height_window_pixels - height_overlap_pixels * 2


def test_valid_non_overlapping_window_source():
    width_window_pixels = height_window_pixels = 100
    width_overlap_pixels = height_overlap_pixels = 0

    window_source = WindowSource(
        width_window_pixels=width_window_pixels,
        width_overlap_pixels=width_overlap_pixels,
        height_window_pixels=height_window_pixels,
        height_overlap_pixels=height_overlap_pixels
        )
    
    assert window_source.width_step_pixels == width_window_pixels - width_overlap_pixels * 2
    assert window_source.height_step_pixels == height_window_pixels - height_overlap_pixels * 2


def test_valid_irregular_window_source():
    width_window_pixels = 100
    height_window_pixels = 80
    width_overlap_pixels = 40
    height_overlap_pixels = 30

    window_source = WindowSource(
            width_window_pixels=width_window_pixels,
            width_overlap_pixels=width_overlap_pixels,
            height_window_pixels=height_window_pixels,
            height_overlap_pixels=height_overlap_pixels
            )

    assert window_source.width_step_pixels == width_window_pixels - width_overlap_pixels * 2
    assert window_source.height_step_pixels == height_window_pixels - height_overlap_pixels * 2

    
def test_too_much_overlap_window_source():
    width_window_pixels = height_window_pixels = 100    
    width_overlap_pixels = height_overlap_pixels = 50

    with pytest.raises(ValueError):
        window_source = WindowSource(
            width_window_pixels=width_window_pixels,
            width_overlap_pixels=width_overlap_pixels,
            height_window_pixels=height_window_pixels,
            height_overlap_pixels=height_overlap_pixels
        )

def test_too_much_width_overlap_window_source():
    width_window_pixels = height_window_pixels = 100    
    width_overlap_pixels = 50
    height_overlap_pixels = 49
    
    with pytest.raises(ValueError):
        window_source = WindowSource(
                width_window_pixels=width_window_pixels,
                width_overlap_pixels=width_overlap_pixels,
                height_window_pixels=height_window_pixels,
                height_overlap_pixels=height_overlap_pixels
            )

def test_too_much_height_overlap_window_source():
    width_window_pixels = height_window_pixels = 100    
    width_overlap_pixels = 49
    height_overlap_pixels = 50
    
    with pytest.raises(ValueError):
        window_source = WindowSource(
                width_window_pixels=width_window_pixels,
                width_overlap_pixels=width_overlap_pixels,
                height_window_pixels=height_window_pixels,
                height_overlap_pixels=height_overlap_pixels
            )


def test_invalid_types_window_source():
    width_window_pixels = height_window_pixels = 100    
    width_overlap_pixels = height_overlap_pixels = 20.0

    with pytest.raises(ValueError):
        window_source = WindowSource(
            width_window_pixels=width_window_pixels,
            width_overlap_pixels=width_overlap_pixels,
            height_window_pixels=height_window_pixels,
            height_overlap_pixels=height_overlap_pixels
        )


def test_overlap_bigger_than_window_window_source():
    width_window_pixels = height_window_pixels = 100    
    width_overlap_pixels = height_overlap_pixels = 110

    with pytest.raises(ValueError):
        window_source = WindowSource(
                    width_window_pixels=width_window_pixels,
                    width_overlap_pixels=width_overlap_pixels,
                    height_window_pixels=height_window_pixels,
                    height_overlap_pixels=height_overlap_pixels
                )
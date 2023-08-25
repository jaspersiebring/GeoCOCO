import numpy as np
from rasterio.windows import Window
from geococo.utils import generate_window_offsets, window_factory, generate_window_polygon
from geococo.models import WindowSource


def test_generate_window_offsets():
    col_off = row_off = 0
    parent_window_width = parent_window_height = 1000
    width_window_pixels = height_window_pixels = 100
    width_overlap_pixels = height_overlap_pixels = 20

    window_source = WindowSource(
        width_window_pixels=width_window_pixels,
        width_overlap_pixels=width_overlap_pixels,
        height_window_pixels=height_window_pixels,
        height_overlap_pixels=height_overlap_pixels
        )
    
    parent_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=parent_window_width,
        height=parent_window_height
        )
    
    offsets = generate_window_offsets(parent_window, window_source)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= parent_window.width + parent_window.col_off - window_source.width_overlap_pixels)
    assert np.all(offsets[:, 1] <= parent_window.height + parent_window.row_off - window_source.height_overlap_pixels)
    assert np.all(offsets >= 0)

    # check if the number of offsets is correct
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / window_source.width_step_pixels)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / window_source.height_step_pixels)
    assert offsets.shape[0] == n_width_steps * n_height_steps
        

def test_generate_window_offsets_equal_window():
    col_off = row_off = 0
    parent_window_width = parent_window_height = 100
    width_window_pixels = height_window_pixels = 100
    width_overlap_pixels = height_overlap_pixels = 0

    window_source = WindowSource(
        width_window_pixels=width_window_pixels,
        width_overlap_pixels=width_overlap_pixels,
        height_window_pixels=height_window_pixels,
        height_overlap_pixels=height_overlap_pixels
        )
    
    parent_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=parent_window_width,
        height=parent_window_height
        )
    
    offsets = generate_window_offsets(parent_window, window_source)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= parent_window.width + parent_window.col_off - window_source.width_overlap_pixels)
    assert np.all(offsets[:, 1] <= parent_window.height + parent_window.row_off - window_source.height_overlap_pixels)
    assert np.all(offsets >= 0)

    # check if the number of offsets is correct
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / window_source.width_step_pixels)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / window_source.height_step_pixels)
    assert offsets.shape[0] == n_width_steps * n_height_steps
        
    
def test_generate_window_offsets_larger_window():
    col_off = row_off = 0
    parent_window_width = parent_window_height = 100
    width_window_pixels = height_window_pixels = 1000
    width_overlap_pixels = height_overlap_pixels = 0

    window_source = WindowSource(
        width_window_pixels=width_window_pixels,
        width_overlap_pixels=width_overlap_pixels,
        height_window_pixels=height_window_pixels,
        height_overlap_pixels=height_overlap_pixels
        )
    
    parent_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=parent_window_width,
        height=parent_window_height
        )
    
    offsets = generate_window_offsets(parent_window, window_source)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= parent_window.width + parent_window.col_off - window_source.width_overlap_pixels)
    assert np.all(offsets[:, 1] <= parent_window.height + parent_window.row_off - window_source.height_overlap_pixels)
    assert np.all(offsets >= 0)

    # check if the number of offsets is correct
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / window_source.width_step_pixels)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / window_source.height_step_pixels)
    assert offsets.shape[0] == n_width_steps * n_height_steps
    

#TODO
# add test for window_factory that checks if boundless leads to non-int vals 

def test_window_factory_len():
    parent_window = Window(0, 0, 1000, 1000)
    window_source = WindowSource(100, 0, 100, 0)
    boundless = False
    windows = list(window_factory(parent_window=parent_window, window_source=window_source, boundless=boundless))
    
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / window_source.width_step_pixels)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / window_source.height_step_pixels)
    assert len(windows) == n_width_steps * n_height_steps

def test_window_factory_boundless_clipping():
    parent_window = Window(0, 0, 100, 100)
    window_source = WindowSource(110, 0, 110, 0)
    boundless = True
    windows = list(window_factory(parent_window=parent_window, window_source=window_source, boundless=boundless))
    assert windows[0].width == 110
    assert windows[0].height == 110

    parent_window = Window(0, 0, 100, 100)
    window_source = WindowSource(110, 0, 110, 0)
    boundless = False
    windows = list(window_factory(parent_window=parent_window, window_source=window_source, boundless=boundless))
    assert windows[0].width == 100
    assert windows[0].height == 100
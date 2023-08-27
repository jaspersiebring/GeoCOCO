import numpy as np
from rasterio.windows import Window
from geococo.utils import generate_window_offsets, window_factory, mask_to_rle, rle_to_mask
from geococo.window_schema import WindowSchema


def test_generate_window_offsets():
    col_off = row_off = 0
    parent_window_width = parent_window_height = 1000
    width_window = height_window = 100
    width_overlap = height_overlap = 20

    schema = WindowSchema(
        width_window=width_window,
        width_overlap=width_overlap,
        height_window=height_window,
        height_overlap=height_overlap
        )
    
    parent_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=parent_window_width,
        height=parent_window_height
        )
    
    offsets = generate_window_offsets(parent_window, schema)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= parent_window.width + parent_window.col_off - schema.width_overlap)
    assert np.all(offsets[:, 1] <= parent_window.height + parent_window.row_off - schema.height_overlap)
    assert np.all(offsets >= 0)

    # check if the number of offsets is correct
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / schema.width_step)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / schema.height_step)
    assert offsets.shape[0] == n_width_steps * n_height_steps
        

def test_generate_window_offsets_equal_window():
    col_off = row_off = 0
    parent_window_width = parent_window_height = 100
    width_window = height_window = 100
    width_overlap = height_overlap = 0

    schema = WindowSchema(
        width_window=width_window,
        width_overlap=width_overlap,
        height_window=height_window,
        height_overlap=height_overlap
        )
    
    parent_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=parent_window_width,
        height=parent_window_height
        )
    
    offsets = generate_window_offsets(parent_window, schema)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= parent_window.width + parent_window.col_off - schema.width_overlap)
    assert np.all(offsets[:, 1] <= parent_window.height + parent_window.row_off - schema.height_overlap)
    assert np.all(offsets >= 0)

    # check if the number of offsets is correct
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / schema.width_step)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / schema.height_step)
    assert offsets.shape[0] == n_width_steps * n_height_steps
        
    
def test_generate_window_offsets_larger_window():
    col_off = row_off = 0
    parent_window_width = parent_window_height = 100
    width_window = height_window = 1000
    width_overlap = height_overlap = 0

    schema = WindowSchema(
        width_window=width_window,
        width_overlap=width_overlap,
        height_window=height_window,
        height_overlap=height_overlap
        )
    
    parent_window = Window(
        col_off=col_off,
        row_off=row_off,
        width=parent_window_width,
        height=parent_window_height
        )
    
    offsets = generate_window_offsets(parent_window, schema)

    # Check that the offsets are within bounds (i.e. 0 < n < parent_window)
    assert np.all(offsets[:, 0] <= parent_window.width + parent_window.col_off - schema.width_overlap)
    assert np.all(offsets[:, 1] <= parent_window.height + parent_window.row_off - schema.height_overlap)
    assert np.all(offsets >= 0)

    # check if the number of offsets is correct
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / schema.width_step)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / schema.height_step)
    assert offsets.shape[0] == n_width_steps * n_height_steps
    

#TODO
# add test for window_factory that checks if boundless leads to non-int vals 

def test_window_factory_len():
    parent_window = Window(0, 0, 1000, 1000)
    schema = WindowSchema(100, 0, 100, 0)
    boundless = False
    windows = list(window_factory(parent_window=parent_window, schema=schema, boundless=boundless))
    
    n_width_steps = np.ceil((parent_window.width + parent_window.col_off) / schema.width_step)
    n_height_steps = np.ceil((parent_window.height + parent_window.row_off) / schema.height_step)
    assert len(windows) == n_width_steps * n_height_steps

def test_window_factory_boundless_clipping():
    parent_window = Window(0, 0, 100, 100)
    schema = WindowSchema(110, 0, 110, 0)
    boundless = True
    windows = list(window_factory(parent_window=parent_window, schema=schema, boundless=boundless))
    assert windows[0].width == 110
    assert windows[0].height == 110

    parent_window = Window(0, 0, 100, 100)
    schema = WindowSchema(110, 0, 110, 0)
    boundless = False
    windows = list(window_factory(parent_window=parent_window, schema=schema, boundless=boundless))
    assert windows[0].width == 100
    assert windows[0].height == 100
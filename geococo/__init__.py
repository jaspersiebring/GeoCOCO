import warnings
from rasterio.errors import NotGeoreferencedWarning

# We make this specific warning 'catchable' to ensure that all rasterio clips are valid
warnings.filterwarnings('error', 'shapes are outside bounds of raster. Are they in different coordinate reference systems?', category=UserWarning)

# TODO: not matching yet (needed for raster_factory)
warnings.filterwarnings('error', 'The given matrix is equal to Affine.identity or its flipped counterpart. GDAL may ignore this matrix and save no geotransform without raising an error. This behavior is somewhat driver-specific.', category=NotGeoreferencedWarning)

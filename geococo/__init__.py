import warnings

# We make this specific warning 'catchable' to ensure that all rasterio clips are valid
warnings.filterwarnings('error', 'shapes are outside bounds of raster. Are they in different coordinate reference systems?', category=UserWarning)

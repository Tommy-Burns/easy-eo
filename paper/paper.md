# Statement of need

Most Earth Observation data no longer arrives as a file on disk. Sentinel-2 and Landsat sit in cloud catalogs, indexed by STAC [@stac_spec] and stored as
cloud-optimized GeoTIFFs. A researcher asking about one place over one season has to query a catalog, pull a small window out of scenes whose cumulative size can go up to hundreds
of megabytes, and line up bands delivered at different resolutions before computing anything at all.

Python has good tools for each step. Rasterio [@rasterio] reads and writes over GDAL [@gdal], NumPy [@numpy] does the arithmetic, GeoPandas [@geopandas] handles the vector files. Joining them up is left to the analyst, and that is where the quiet failures happen. `nodata` has to be masked before a mean, or fill values land in the statistic. Integer bands have to be promoted before a subtraction to prevent overflow and underflow errors.

The usual way out is a hosted platform. Google Earth Engine [@gorelick2017] is the obvious one and it is very good at scale. But it needs an account, the work runs on someone else's servers, and the analysis has to be written in its deferred API. That is a poor fit for a study which has to run locally and still reproduce in three years.

Easy-EO covers the whole path. A search against any STAC catalog returns scenes; loading one fetches only the window over the area of interest, over HTTP; the masking and promotion rules apply unless you turn them off. For anything it does not cover, the rasterio dataset underneath is exposed directly, and the pixels can be accessed as a NumPy array or as a georeferenced xarray DataArray [@xarray].

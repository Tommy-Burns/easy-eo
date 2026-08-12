# Statement of need

Most Earth Observation data no longer arrives as a file on disk. Sentinel-2 and Landsat sit in cloud catalogs, indexed by STAC [@stac_spec] and stored as
cloud-optimized GeoTIFFs. A researcher asking about one place over one season has to query a catalog, pull a small window out of scenes whose cumulative size can go up to hundreds
of megabytes, and line up bands delivered at different resolutions before computing anything at all.

Python has good tools for each step. Rasterio [@rasterio] reads and writes over GDAL [@gdal], NumPy [@numpy] does the arithmetic, GeoPandas [@geopandas] handles the vector files. Joining them up is left to the analyst, and that is where the quiet failures happen. `nodata` has to be masked before a mean, or fill values land in the statistic. Integer bands have to be promoted before a subtraction to prevent overflow and underflow errors.

The usual way out is a hosted platform. Google Earth Engine [@gorelick2017] is the obvious one and it is very good at scale. But it needs an account, the work runs on someone else's servers, and the analysis has to be written in its deferred API. That is a poor fit for a study which has to run locally and still reproduce in three years.

Easy-EO covers the whole path. A search against any STAC catalog returns scenes; loading one fetches only the window over the area of interest, over HTTP; the masking and promotion rules apply unless you turn them off. For anything it does not cover, the rasterio dataset underneath is exposed directly, and the pixels can be accessed as a NumPy array or as a georeferenced xarray DataArray [@xarray].

# State of the field

rastereasy [@corpetti2025] is the nearest neighbour: a chainable image object over rasterio, aimed at the same non-specialist audience. It works on imagery the user already holds, and extends from there into filtering, dimensionality reduction and classifier fitting. Easy-EO extends in the other direction. A search against any STAC catalog returns matching scenes, and loading one pulls only the pixels under the area of interest across the network, so a study site can be analysed without the tile it sits in ever being downloaded entirely. Band names survive the round-trip to disk, and a band can be addressed as `"red"` anywhere an index is accepted. The `nodata` and dtype rules hold on every operation, whether or not the analyst thinks about them. Neither package is a superset of the other.

The rest of the field sits further off. rioxarray [@rioxarray] over xarray [@xarray] is the better tool when the work is genuinely multidimensional, a long time series or a chunked computation under Dask; Easy-EO hands off to it at the boundary. xarray-spatial [@xarray_spatial] adds algorithms to that model but leaves data access to the caller, and eo-learn [@eolearn] targets machine-learning pipelines through its heavier EOPatch abstraction. EarthPy [@earthpy] offers helper functions for stacking and plotting rasters, which stand alone: there is no dataset object to chain from, and data access is left to the user.

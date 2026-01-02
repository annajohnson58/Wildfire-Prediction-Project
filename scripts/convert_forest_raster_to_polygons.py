import geopandas as gpd
import rasterio
from rasterio.features import shapes
import numpy as np
import os

RASTER_PATH = "data/shapefiles/Kerala_Forest_Mask.tif"
OUTPUT_PATH = "data/shapefiles/kerala_forests.geojson"

def main():
    with rasterio.open(RASTER_PATH) as src:
        mask = src.read(1)
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for s, v in shapes(mask, mask=mask > 0, transform=src.transform)
        )

    geoms = list(results)
    gdf = gpd.GeoDataFrame.from_features(geoms)
    gdf = gdf[gdf['value'] == 1]  # forest only

    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print("✅ Saved forest polygons to", OUTPUT_PATH)

if __name__ == "__main__":
    main()

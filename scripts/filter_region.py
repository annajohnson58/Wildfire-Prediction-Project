import osmnx as ox
import geopandas as gpd

# Define Kerala boundary using place name
kerala = ox.geocode_to_gdf("Kerala, India")

# Query forest polygons from OSM
tags = {"landuse": "forest"}
forests = ox.features_from_place("Kerala, India", tags)

# Filter and clean
forests = forests[forests.geometry.type.isin(["Polygon", "MultiPolygon"])]
forests = forests.set_crs(kerala.crs)

# Clip to Kerala boundary
forests = gpd.overlay(forests, kerala, how="intersection")

# Save to shapefile
forests.to_file("data/shapefiles/kerala_osm_forests.shp")
print("✅ Forest shapefile saved: kerala_osm_forests.shp")

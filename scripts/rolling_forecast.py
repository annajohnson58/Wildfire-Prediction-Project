import geopandas as gpd

forest = gpd.read_file("data/shapefiles/kerala_osm_forests.shp")

# Use the English name column
unique_forests = forest["name_en"].dropna().unique()

print("✅ Number of forests included:", len(unique_forests))
print("\n📋 Forests covered in the project:")
for f in unique_forests:
    print("-", f)

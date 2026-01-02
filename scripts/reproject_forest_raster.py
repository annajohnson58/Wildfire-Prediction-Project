import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

src_path = "data/shapefiles/Kerala_Forest_Mask.tif"
dst_path = "data/shapefiles/Kerala_Forest_Mask_4326.tif"

with rasterio.open(src_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        "crs": "EPSG:4326",
        "transform": transform,
        "width": width,
        "height": height
    })

    with rasterio.open(dst_path, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest
            )

print("✅ Reprojected raster saved to", dst_path)

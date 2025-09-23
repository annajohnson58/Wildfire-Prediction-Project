import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 📥 Load Kerala shapefile and predicted risk scores
kerala = gpd.read_file("data/shapefiles/kerala_districts.shp").to_crs("EPSG:4326")
risk = pd.read_csv("data/fused/predicted_risk.csv")

# 🔗 Clean and merge
kerala['district'] = kerala['DISTRICT'].str.strip().str.lower()
risk['district'] = risk['district'].str.strip().str.lower()
kerala = kerala.merge(risk, on='district', how='left')

# 🎨 Plot fire risk map
fig, ax = plt.subplots(figsize=(10, 12))
kerala.plot(column='predicted_risk',
            cmap='OrRd',
            linewidth=0.8,
            edgecolor='black',
            legend=True,
            ax=ax)

ax.set_title("🔥 Predicted Fire Risk Across Kerala", fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()

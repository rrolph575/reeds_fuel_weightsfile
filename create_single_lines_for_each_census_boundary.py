

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/rrolph/Desktop/ReEDS-2.0")
import reeds 
from shapely.geometry import LineString

""" # Commented because this is a manual visualization file and don't want to accidentally run


# Load census division boundaries
census_boundaries_gdf = reeds.io.get_dfmap()['cendiv']
country_boundaries_gdf = reeds.io.get_dfmap()['country']
# Combine all Census Division geometries
boundaries = census_boundaries_gdf.boundary
country_boundaries = country_boundaries_gdf.boundary

# Ensure CRS match
assert country_boundaries_gdf.crs == boundaries.crs, "CRS mismatch! Reproject before proceeding."

# Remove near overlapping areas between boundaries and country_boundaries
boundaries_gdf = gpd.GeoDataFrame(geometry=boundaries, crs=boundaries.crs)
country_boundaries_gdf = gpd.GeoDataFrame(geometry=country_boundaries, crs=country_boundaries.crs)
census_without_country = gpd.overlay(boundaries_gdf, country_boundaries_gdf, how='difference')


idx_num = 8
fig, ax = plt.subplots(figsize=(10, 8))
row = census_without_country.iloc[idx_num]

# Check if the geometry is a MultiLineString
if row.geometry.geom_type == "MultiLineString":
    # Convert to GeoSeries for plotting
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color='lightblue', edgecolor='black')
else:
    # Plot directly if it's not a MultiLineString
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color='lightblue', edgecolor='black')

# Add a label for the geometry using the row index
label_point = row.geometry.representative_point()  # Get a point guaranteed to be inside the geometry
ax.text(label_point.x, label_point.y, str(idx_num), fontsize=8, ha='center', color='red')

plt.title('Census Boundary')
plt.show()



## If the plot above shows more than one census boundary then trim it down to 1
# Extract the geometry for the given row
row = census_without_country.iloc[idx_num]
geometry = row.geometry

# Check if the geometry is a MultiLineString
if geometry.geom_type == "MultiLineString":
    # Extract the first line from the MultiLineString
    single_line = list(geometry.geoms)[1]  # Take the first LineString
else:
    # If it's not a MultiLineString, use the geometry directly
    single_line = geometry

# Plot the single line
fig, ax = plt.subplots(figsize=(10, 8))
gpd.GeoSeries([single_line]).plot(ax=ax, color='blue', linewidth=2)

plt.title(f"Single Line from Region {idx_num}")
plt.show()


# Create a new GeoDataFrame with specified columns
new_gdf = gpd.GeoDataFrame(columns=['geometry', 'region_name'], crs=census_without_country.crs)


## Edit and run this after manually inspecting each index above 
# Assign values to the first row
new_gdf.loc[9, 'geometry'] = row.geometry  # Assign the geometry
#new_gdf.loc[4, 'geometry'] = single_line # comment/uncomment this one and line above based on manual inspection above
new_gdf.loc[9, 'region_name'] = 'West_South_Central'  # Assign the region name

print(new_gdf)


## Once completed manually, save file
new_gdf.to_file('census_boundaries_individual.geojson', driver='GeoJSON')

# Load the newly created GeoJSON file to verify
new_gdf_loaded = gpd.read_file('census_boundaries_individual.geojson')


"""

"""
This script creates a weights.csv file for any given input shapefile that contains p regions.
"""


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/rrolph/Desktop/ReEDS-2.0")
import reeds 
from shapely.geometry import LineString
from shapely.ops import nearest_points, unary_union, polygonize
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


#%%
## Define paths
reedspath = '~/Desktop/ReEDS-2.0/'
ifile_shp_p_regions = reedspath + 'inputs/shapefiles/US_PCA/US_PCA.shp'  # p regions
# county shapefiles
ifile_shp = reedspath + 'inputs/shapefiles/US_PCA/US_PCA.shp'
#ifile_shp = "C:/Users/rrolph/Desktop/ReEDS-2.0/inputs/shapefiles/US_COUNTY_2022/US_COUNTY_2022.shp"
# gridded shapefile, supply curve from reV
#ifile_shp = "C:/Users/rrolph/OneDrive - NREL/Projects/FY25/ReEDS_development/weights/supply_curve_grid128_prjusalb/supply_curve_grid128_prjusalb.shp"

ifile_weights = reedspath + 'inputs/fuelprices/cendivweights.csv' # this is Donna's file. This is what the output should resemble
distance_btwn_centroids = 'transmission_distance_cost_500kVdc_ba.csv'
# Load census division boundaries
census_boundaries_gdf = reeds.io.get_dfmap()['cendiv']
country_boundaries_gdf = reeds.io.get_dfmap()['country']
# Load distances between each centroid of the p-regions
df_centroid_dist = pd.read_csv('transmission_distance_cost_500kVdc_ba.csv')



#%%  Select region to plot
region_name = 'East_South_Central' 

## Load files 
gdf_p_regions = gpd.read_file(ifile_shp_p_regions) # shapefile
gdf = gpd.read_file(ifile_shp) # p regions
df = pd.read_csv(ifile_weights) # weights file from Donna H. (trying to reproduce in this script)
#df = df[df['r'] <= 'p134'] # p-regions here are numbered higher than 134 and still in the US..
df = df[df['MEX'] != 1.1] # Remove p regions where MEX = 1.1 
dist_df = pd.read_csv(distance_btwn_centroids)

## Plot shapefile
gdf.plot(figsize=(10,6), edgecolor='black')
plt.show()


#%%  Map df to gdf to plot the weights
# p regions are given in gdf with key 'rb'
# p regions are given in df with key 'r'
# Merge the DataFrame with the GeoDataFrame on 'r' and 'rb'
merged_gdf = gdf_p_regions.merge(df, left_on='rb', right_on='r', how='left')
# Plot the weights
region_name = 'East_South_Central'
fig, ax = plt.subplots()
merged_gdf.plot(column=region_name, ax=ax, legend=True, vmin=0, vmax=1)
#merged_gdf.plot(column=region_name, ax=ax, legend=True)
# Add census boundaries
census_boundaries_gdf.boundary.plot(ax=ax, color='red')
plt.title(f"{region_name}, Original weight file")


#%% Define p-regions within the western interconnect
df_state_p_region_def = gdf_p_regions.groupby('st')['rb'].apply(list).reset_index()
df_state_p_region_def.columns = ['st', 'p_regions_list']
# tx is [p48, p57, p59, p60, p61, p62, p63, p64, p65, p66, p67]
#df_states_nom_region_df = pd.DataFrame(columns=df.columns)
#df_states_nom_region_df = df_states_nom_region_df.drop(columns=['MEX', 'r'])
regions = {
    'East_North_Central': ['wi', 'il', 'in', 'mi', 'oh'],  ## put in states that are in same syntax as df_state_p_region_def
    'East_South_Central': ['ms', 'al', 'tn', 'ky'],
    'Mid_Atlantic': ['ny','pa','nj'],
    'Mountain': ['mt','wy','id','co','ut','nv','az','nm'],
    'New_England': ['me', 'nh','vt', 'ma', 'ri', 'ct'],
    'Pacific': ['wa', 'or', 'ca'],
    'South_Atlantic': ['wv', 'de', 'md', 'va', 'nc', 'sc', 'ga', 'fl'],
    'West_North_Central': ['nd', 'sd', 'ne', 'ks', 'mn', 'ia', 'mo'],
    'West_South_Central': ['ok', 'tx', 'ar', 'la']
}

mountain_states = regions['Mountain']
pacific_states = regions['Pacific']
# Combine the two lists of states
states_to_filter = mountain_states + pacific_states
# Filter the DataFrame for these states
filtered_df = df_state_p_region_def[df_state_p_region_def['st'].isin(states_to_filter)]
p_regions_in_western_interconnect = filtered_df['p_regions_list'].explode().tolist()
p_regions_in_western_interconnect.extend(['p32', 'p59'])
p_regions_in_western_interconnect.remove('p35')
p_regions_in_western_interconnect.remove('p47')



#%% Find p regions in Mountain and West_North_Central 
## Find p regions in Mountain
mountain_gdf = census_boundaries_gdf.loc[['Mountain']]['geometry']
# Ensure both GeoDataFrames use the same CRS
gdf_p_regions = gdf_p_regions.to_crs(mountain_gdf.crs)
# Create a single geometry for the Mountain region
mountain_geom = mountain_gdf.unary_union
# Filter rows where geometry is contained within Mountain geometry
contained_gdf = gdf_p_regions[gdf_p_regions.geometry.within(mountain_geom)]
p_regions_in_mountain = contained_gdf['rb'].unique()

## Find p regions in 'West_North_Central'
west_north_central_gdf = census_boundaries_gdf.loc[['West_North_Central']]['geometry']
# Ensure both GeoDataFrames use the same CRS
gdf_p_regions = gdf_p_regions.to_crs(west_north_central_gdf.crs)
# Create a single geometry for the region
wn_central_geom = west_north_central_gdf.unary_union
# Filter rows where geometry is contained within wn geometry
contained_gdf = gdf_p_regions[gdf_p_regions.geometry.within(wn_central_geom)]
p_regions_in_west_north_central = contained_gdf['rb'].unique()


#%%  Plot centroids of the p-regions
# Calculate the centroids of the geometries
gdf['centroid'] = gdf.geometry.centroid
# Extract the x and y coordinates of the centroids
centroids = gdf['centroid']
centroid_x = centroids.x
centroid_y = centroids.y
# Plot the original geometries
ax = gdf.plot(figsize=(10, 10), color='lightblue', edgecolor='black')
# Plot the centroids as red points
ax.scatter(centroid_x, centroid_y, color='red', marker='o', label='Centroid')
# Add labels for each centroid
#for i, row in gdf.iterrows():
#    ax.text(row['centroid'].x, row['centroid'].y, str(row['OBJECTID_1']), color='black', fontsize=8)
# Add a title and legend
plt.title('Centroids of Geometries')
plt.legend()
# Show the plot
plt.show()



# %% Find the max distance, which is the centroid that is furthest from the nearest census boundary 
# Determine the distances from each centroid to the nearest census boundary, without counting country boundary as a census boundary.

# Combine all Census Division geometries
boundaries = census_boundaries_gdf.boundary
country_boundaries = country_boundaries_gdf.boundary

# Ensure CRS match
assert country_boundaries_gdf.crs == boundaries.crs, "CRS mismatch! Reproject before proceeding."

# Remove near overlapping areas between boundaries and country_boundaries
boundaries_gdf = gpd.GeoDataFrame(geometry=boundaries, crs=boundaries.crs)
country_boundaries_gdf = gpd.GeoDataFrame(geometry=country_boundaries, crs=country_boundaries.crs)
census_without_country = gpd.overlay(boundaries_gdf, country_boundaries_gdf, how='difference')


# %% Check plot
fig, ax = plt.subplots(figsize=(10, 8))
# Plot boundaries
#boundaries.plot(ax=ax, color='red', alpha=0.5, label='Census Boundaries')
country_boundaries.plot(ax=ax, color='green', alpha=0.5, label='Country Boundaries', linestyle='--')
census_without_country.plot(ax=ax, color='blue', alpha=0.5, label='Boundaries without Country line')
# Add legend
ax.legend()
# Add title
ax.set_title("Census Boundaries, removed Country Boundaries")
# Show the plot
plt.show()
# Ensure CRS match
assert gdf.crs == census_without_country.crs, "CRS mismatch! Reproject before proceeding."



# %% Plot the indices of the census boundaries to select the right ones for nearest distance
fig, ax = plt.subplots(figsize=(10, 8))
census_without_country.plot(ax=ax, color='lightblue', edgecolor='black')
# Add labels for each geometry using the row index
for idx, row in census_without_country.iterrows():
#for idx, row in census_without_country.iloc[7]:
    label_point = row.geometry.representative_point()  # Get a point guaranteed to be inside the geometry
    ax.text(label_point.x, label_point.y, str(idx), fontsize=8, ha='center', color='red')
plt.title('Census Without Country with Row Index Labels')
plt.show()


# %% Plot only a specified index of the census boundary
idx_num = 3
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



#%% Calculate the distance from each centroid to the nearest census boundary

# !!! change the below to only calculate from one census boundary.. the one in the current loop.

gdf_each_census_border_separated = gpd.read_file('census_boundaries_individual.geojson') # Made by create_single_lines_for_each_region.py
gdf_each_census_border_separated.plot(column = 'region_name')



# country_border = country_boundaries_gdf.geometry.iloc[0]




# Initialize empty DataFrames for distances and lines
distances_df = pd.DataFrame(index=gdf.index, columns=list(census_boundaries_gdf.index) + ['rb'])
lines_df = pd.DataFrame(index=gdf.index, columns=list(census_boundaries_gdf.index) + ['rb'])

for region in census_boundaries_gdf.index.to_list():
    for idx, row in gdf.iterrows():
        centroid = row['centroid']
        rb_value = row['rb']


        # Determine the census boundary to use for the current region
        census_boundary_to_use = gdf_each_census_border_separated[
            gdf_each_census_border_separated['region_name'] == region
        ].geometry.values[0]

        # Find the nearest point and calculate the distance
        nearest_point = nearest_points(centroid, census_boundary_to_use)[1]
        distance = centroid.distance(nearest_point)

        # Handle single or multiple distances
        if isinstance(distance, float):  # Single distance case
            line = LineString([centroid, nearest_point])
            distances_df.loc[idx, region] = distance
            distances_df.loc[idx, 'rb'] = rb_value
            lines_df.loc[idx, region] = line
            lines_df.loc[idx, 'rb'] = rb_value
        else:  # Multiple distances case
            idx_min = distance.idxmin()
            nearest_single_point = nearest_point.iloc[idx_min].geometry
            line = LineString([centroid, nearest_single_point])
            distances_df.loc[idx, region] = distance.iloc[idx_min]
            distances_df.loc[idx, 'rb'] = rb_value
            lines_df.loc[idx, region] = line
            lines_df.loc[idx, 'rb'] = rb_value



## Plot distances_df on a map
# rename 'cendiv' in distances to p reigons
merged_gdf = gdf.merge(distances_df, left_on='rb', right_on='rb', how='left')
region_name = 'Mountain'

fig, ax = plt.subplots()
# Plot the data with the specified colormap
plot = merged_gdf.plot(column=region_name, ax=ax, legend=False, cmap='viridis')
# Create a ScalarMappable using the same colormap and normalization as the plot
# Overlay the census boundaries
census_boundaries_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=merged_gdf[region_name].min(), vmax=merged_gdf[region_name].max()))
sm._A = []  # Dummy array for ScalarMappable
# Add the colorbar to the plot
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(f'{region_name} Values')


## !!!! 
## Get rid of the appropriate values so that distances are 0 if they cross western interconnect
# Go through each row in merged_gdf and check if the gdf['centroid'] is within the western interconnect. For this, you will have to create an enclosed shapefile west of the western interconnect.

country_border = country_boundaries_gdf.geometry.iloc[0]

# To make western interconnect one line, can try to use this... remove the pacific boundary and then take the points that are only the western most of the resulting linestrings
gdf_each_census_border_separated = gpd.read_file('census_boundaries_individual.geojson') # Made by create_single_lines_for_each_region.py
gdf_each_census_border_separated.plot(column = 'region_name')


from shapely.geometry import LineString
from shapely.ops import unary_union, nearest_points


# Extract the geometries for Mountain and West South Central
mountain_geometry = gdf_each_census_border_separated.loc[
    gdf_each_census_border_separated['region_name'] == 'Mountain', 'geometry'
].values[0]

west_south_central_geometry = gdf_each_census_border_separated.loc[
    gdf_each_census_border_separated['region_name'] == 'West_South_Central', 'geometry'
].values[0]

# Find the closest points between the two geometries
closest_points = nearest_points(mountain_geometry, west_south_central_geometry)
closest_point = closest_points[0]  # The point on Mountain geometry

# Filter out points east of the closest point
filtered_lines = []
for line in mountain_geometry.geoms if mountain_geometry.geom_type == 'MultiLineString' else [mountain_geometry]:
    filtered_coords = [coord for coord in line.coords if coord[0] <= closest_point.x]
    if len(filtered_coords) > 1:  # Ensure valid line remains
        filtered_lines.append(LineString(filtered_coords))

# Create a new GeoDataFrame with the filtered geometry
filtered_geometry = unary_union(filtered_lines)
gdf_filtered = gpd.GeoDataFrame(
    {'region_name': ['Mountain_Filtered'], 'geometry': [filtered_geometry]},
    crs=gdf_each_census_border_separated.crs
)

# Display the new GeoDataFrame
print(gdf_filtered)

# Plot the filtered geometry
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
gdf_filtered.plot(ax=ax, color='blue', edgecolor='black')

# Add title and labels
ax.set_title('Filtered Geometry: Western Interconnect', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.show()


gdf_western_interconnect = gdf_filtered.copy()


### merged_gdf has the distances and also the centroids of each respective p-region.

## Go through each row of merged_gdf, and 
## if centroid of that row is west of gdf_western_interconnect, then make the values of census_boundaries_gdf.index.to_list() minus Mountain and Pacific to 0. 
## if centroid of that row is east of gdf_western_interconnect, then make the values of 'Pacific' and 'Mountain' to 0.


for idx, row in merged_gdf.iterrows():

    
    # Check if the centroid is west of the western interconnect
    centroid = row['centroid']
    
    # Extract the western interconnect geometry
    western_interconnect_line = gdf_western_interconnect.geometry.iloc[0]
    
    # Find the nearest point on the western interconnect line based on centroid.y
    nearest_point = min(
        western_interconnect_line.coords,
        key=lambda coord: abs(coord[1] - centroid.y)  # Compare y-coordinates
    )
    
    # Get the x-coordinate of the nearest point
    nearest_x = nearest_point[0]
    
    # Check if the centroid is west of the nearest x-coordinate
    if centroid.x <= nearest_x:
        # Set distances to 0 for all regions except Mountain and Pacific
        for region in census_boundaries_gdf.index.to_list():
            if region not in ['Mountain', 'Pacific']:
                merged_gdf.at[idx, region] = 0
    #if centroid.x > gdf_western_interconnect.geometry.bounds.minx.values[0]:
    else:
        # Set distances to 0 for Mountain and Pacific
        merged_gdf.at[idx, 'Mountain'] = 0
        merged_gdf.at[idx, 'Pacific'] = 0


region_name = 'Mountain'

fig, ax = plt.subplots()
# Plot the data with the specified colormap
plot = merged_gdf.plot(column=region_name, ax=ax, legend=False, cmap='viridis')
# Overlay the census boundaries
census_boundaries_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
# Create a ScalarMappable using the same colormap and normalization as the plot
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=merged_gdf[region_name].min(), vmax=merged_gdf[region_name].max()))
sm._A = []  # Dummy array for ScalarMappable
# Add the colorbar to the plot
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(f'{region_name} Values')



region_name = 'East_South_Central'

fig, ax = plt.subplots()
# Plot the data with the specified colormap
plot = merged_gdf.plot(column=region_name, ax=ax, legend=False, cmap='viridis')
# Overlay the census boundaries
census_boundaries_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
# Create a ScalarMappable using the same colormap and normalization as the plot
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=merged_gdf[region_name].min(), vmax=merged_gdf[region_name].max()))
sm._A = []  # Dummy array for ScalarMappable
# Add the colorbar to the plot
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(f'{region_name} Values')









## Now you have to convert distances_df to your weights units and make them 0 if the distances are 
## Set the values to 1 that are more than a threshold of miles from the appropriate census border. 


# Convert distances_df from meters to miles
distances_df_miles = distances_df.copy()

# Apply the conversion factor to all numeric columns (excluding 'rb')
for col in distances_df.columns:
    if col != 'rb':  # Skip the 'rb' column
        distances_df_miles[col] = distances_df[col] * 0.000621371



# Create a new DataFrame where values less than 200 miles are set to 1
weights_new = distances_df_miles.copy()


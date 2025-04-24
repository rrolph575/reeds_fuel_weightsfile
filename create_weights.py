
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
from shapely.ops import nearest_points
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm



## Define paths
reedspath = '~/Desktop/ReEDS-2.0/'
ifile_shp = reedspath + 'inputs/shapefiles/US_PCA/US_PCA.shp'  # p regions
# !! test this C:\Users\rrolph\Desktop\ReEDS-2.0\inputs\shapefiles\US_COUNTY_2022\US_COUNTY_2022.shp

ifile_weights = reedspath + 'inputs/fuelprices/cendivweights.csv' # this is an example of what the output should look like
distance_btwn_centroids = 'transmission_distance_cost_500kVdc_ba.csv'
# Load census division boundaries
census_boundaries_gdf = reeds.io.get_dfmap()['cendiv']
country_boundaries_gdf = reeds.io.get_dfmap()['country']


# Select region to plot
region_name = 'Pacific' 

## Load files 
gdf = gpd.read_file(ifile_shp) # shapefile
df = pd.read_csv(ifile_weights) # weights file from Donna H. (trying to reproduce in this script)
#df = df[df['r'] <= 'p134'] # p-regions here are numbered higher than 134 and still in the US..
df = df[df['MEX'] != 1.1] # Remove p regions where MEX = 1.1 
dist_df = pd.read_csv(distance_btwn_centroids)

## Plot shapefile
gdf.plot(figsize=(10,6), edgecolor='black')
plt.show()


## Map df to gdf to plot the weights
# p regions are given in gdf with key 'rb'
# p regions are given in df with key 'r'
# Merge the DataFrame with the GeoDataFrame on 'r' and 'rb'
merged_gdf = gdf.merge(df, left_on='rb', right_on='r', how='left')

# Plot the weights
#region_name = 'Mountain'
fig, ax = plt.subplots()
merged_gdf.plot(column=region_name, ax=ax, legend=True)
# Add census boundaries
census_boundaries_gdf.boundary.plot(ax=ax, color='red')
plt.title(f"{region_name}, Donna's weight file")


## Our goal is using input gdf and then generate weights df from that. 
### For every centroid, calculate the distance to the closest centroid outside of the census boundary 
# Define centroids within the census boundaries. 
df_state_p_region_def = gdf.groupby('st')['rb'].apply(list).reset_index()
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


# load distances between each centroid of the p-regions
df_centroid_dist = pd.read_csv('transmission_distance_cost_500kVdc_ba.csv')

# Calculate the distance between each centroid to the closest centroid outside of the census boundary 
# but don't let them calculate across the western interconnect, which is defined by p-region
#p_regions_in_western_interconnect = # those p regions within 'Pacific' and 'Mountain' - p35 + p32 -p47 + p59
# Extract states for 'Mountain' and 'Pacific' regions
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


### Plot centroids of the p-regions
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
for i, row in gdf.iterrows():
    ax.text(row['centroid'].x, row['centroid'].y, str(row['OBJECTID_1']), color='black', fontsize=8)
# Add a title and legend
plt.title('Centroids of Geometries')
plt.legend()
# Show the plot
plt.show()



### Find the max distance, which is the centroid that is furthest from the nearest census boundary (not crossing the western interconnect.. use the df above)  (this should probably be the threshold)
# Determine the distances from each centroid to the nearest census boundary, without counting country boundary as a census boundary.

# Combine all Census Division geometries
boundaries = census_boundaries_gdf.boundary
country_boundaries = country_boundaries_gdf.boundary

# Ensure CRS match
assert country_boundaries_gdf.crs == boundaries.crs, "CRS mismatch! Reproject before proceeding."

# Remove near overlapping areas between boundaries and country_boundaries

# Convert boundaries to a GeoDataFrame if it's not already
boundaries_gdf = gpd.GeoDataFrame(geometry=boundaries, crs=boundaries.crs)
country_boundaries_gdf = gpd.GeoDataFrame(geometry=country_boundaries, crs=country_boundaries.crs)
# Remove overlapping areas
census_without_country = gpd.overlay(boundaries_gdf, country_boundaries_gdf, how='difference')


## Check plot
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

# Calculate the distances from each centroid to its nearest census boundary
distances = []
lines = []
# Loop through each row
for idx, row in gdf.iterrows():
    centroid = row['centroid']
    nearest_point = nearest_points(centroid, census_without_country)[1]
    distance = centroid.distance(nearest_point)
    idx_min = distance.idxmin()
    nearest_single_point = nearest_point.iloc[idx_min].geometry
    line = LineString([centroid, nearest_single_point.iloc[0]])

    distances.append(distance.iloc[idx_min].values[0][0])
    lines.append(line)

# Add results to gdf
gdf['dist_to_boundary'] = distances
gdf['dist_line'] = lines
# Convert to GeoDataFrame for plotting
lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs) 



### Find the max distance within each census boundary
# Make a list of the p-regions within each census boundary
# Ensure both are using the same CRS
gdf = gdf.to_crs(census_boundaries_gdf.crs)
# Rename gdf['geometry'] to gdf['geom_of_p_regions']
gdf['geom_of_p_regions'] = gdf['geometry']
# Spatial join (using 'centroid' of gdf for point-in-polygon test)
gdf['geometry'] = gdf['centroid']  # Temporarily use centroid as geometry
joined = gpd.sjoin(gdf, census_boundaries_gdf, how='inner', predicate='within')
# Group by 'cendiv' (index of census_boundaries_gdf) and list 'rb'
p_regions_by_cendiv = joined.groupby('index_right')['rb'].apply(list)
# Create a mapping for 'p' values to their respective regions
p_to_region = {}
for region, p_values in p_regions_by_cendiv.items():
    for p in p_values:
        p_to_region[p] = region
# Add a new column to gdf that maps 'rb' to the region name
gdf['region'] = gdf['rb'].map(p_to_region)
# Group by 'region' and find the row with the maximum 'dist_to_boundary' for each region
max_dist_rows = gdf.loc[gdf.groupby('region')['dist_to_boundary'].idxmax()]



## Plot the distances and census boundary.
fig, ax = plt.subplots(figsize=(12, 10))
# Plot base polygons
gdf.plot(ax=ax, edgecolor='black', facecolor='lightgray')
# Plot centroids
gdf.set_geometry('centroid').plot(ax=ax, color='red', markersize=10, label='Centroids')
# Plot boundary lines
census_without_country.plot(ax=ax, color='blue', linewidth=1, label='Boundaries')
# Plot connection lines
#lines_gdf.plot(ax=ax, color='green', linewidth=1, label='Distance Lines')
# Plot maximum distance lines
dist_lines_gdf = gpd.GeoDataFrame(max_dist_rows.copy(), geometry='dist_line')
dist_lines_gdf.plot(ax=ax, color='darkred', linewidth=8, label='Max Distance Line')
# Overlay the original point locations
max_dist_rows.plot(ax=ax, color='red', markersize=50, label='Max Distance Point')
# Add legend and formatting
plt.title('Distance from farthest centroid to closest census division boundary', fontsize=18)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()



### If the distance is over a certain threshold, then the weight for that p-region is a 1 because it is well-within the census region (defined at top of script)
# Group by region and compute the threshold
region_thresholds = dist_lines_gdf.groupby('region')['dist_to_boundary'].max()
# Map these thresholds back to the DataFrame
dist_lines_gdf['threshold_for_weight_1'] = dist_lines_gdf['region'].map(region_thresholds)
# Merge based on 'rb' column
gdf = gdf.merge(
    dist_lines_gdf[['region', 'threshold_for_weight_1']],
    on='region',
    how='left'  # Use 'left' to keep all rows in gdf
)

#Assign weights of 1 for each p region that is over this threshold !! This will have to be normalized later across neighboring regions.
# Merge dist_lines_gdf with gdf on 'rb' (left join to retain all rows from gdf)
gdf['weight'] = (gdf['dist_to_boundary'] == gdf['threshold_for_weight_1']).astype(int)


### Plot the weights of 1 to make sure it assigned the correct center p-regions
# Plot the weights
fig, ax = plt.subplots()
gdf['weight']=gdf['weight'].fillna(0) # NaN caused weird colormapping issues
gdf.set_geometry('geom_of_p_regions').plot(column='weight', ax=ax, legend=True, cmap='viridis', edgecolor='orange')
# Add census boundaries
census_without_country.plot(ax=ax, color='red')
plt.title('Center weighted p-regions')



### Decrease the weights for the other p-regions in that census division as the distance to the census boundary decreases
# Create a new weights df 
# List of region columns
regions = [
    "East_North_Central", "East_South_Central", "Mid_Atlantic",
    "Mountain", "New_England", "Pacific", "South_Atlantic",
    "West_North_Central", "West_South_Central"]
# Create the 'r' column with values p1 to p156
r_values = [f"p{i}" for i in range(1, 157)]
# Create a DataFrame with zeros
weights_df = pd.DataFrame(0, index=range(156), columns=regions)
weights_df.insert(0, 'r', r_values)



# Create weights df and assign 1 to the center p-regions based on above
regions = gdf['region'].dropna().unique().tolist()
merged = weights_df.merge(gdf[['rb', 'region', 'weight']], left_on='r', right_on='rb', how='left')
for region in regions:
    condition = (merged['region'] == region) & (merged['weight'] == 1.0)
    merged.loc[condition, region] = 1.0
# Fill missing region columns with 0 for now
merged[regions] = merged[regions].fillna(0.0)
# Update the weights_df with the original columns + new region columns
df_weights_updated = merged[weights_df.columns.tolist()]



## Check the new weights are applied to the correct middle areas
merged_gdf = gdf.merge(df_weights_updated, left_on='rb', right_on='r', how='left')  
# Plot the weights
#region_name = 'Mountain'  # Change to 'Pacific', 'Mountain', etc. 
fig, ax = plt.subplots()
merged_gdf.set_geometry('geom_of_p_regions').plot(column=region_name, ax=ax, legend=True)
# Add census boundaries
census_boundaries_gdf.boundary.plot(ax=ax, color='red')
plt.title(region_name)
# Also did a manual check where p29 is 1 for mountain in both gdf and df_weights_updated.


### Above checks out.. now we have to fill in the rest of the columns so the weights linearly decrease with increasing distance from the p regions with value 1.  (don't replace the values in the df_weights_updated if the value is already 1.)  Can't use the distance to the boundaries here because the weights values have to cross the boundaries.


## For each region in df_weights_updated
# find the p regions with value 1.
for region in df.keys().drop(['r', 'MEX']).tolist():
#for region in ['Mountain']:
    # Find all 'r' values where region == 1
    r_values_region = df_weights_updated[df_weights_updated[region] == 1]['r']

    # Filter dist_df to get the rows where 'r' matches r_values_region
    filtered_dist_df = dist_df[dist_df['r'].isin(r_values_region)]

    # Get max and min distances to normalize
    min_dist = filtered_dist_df['length_miles'].min()
    max_dist = filtered_dist_df['length_miles'].max()

    # For each row in filtered_dist_df, find the matching row in df_weights_updated and update the 'Pacific' column
    #slope = 2 # the rate at which the 1 weight decreases from center
    coeff = 6
    # Get the min and max values for the target region in df
    #min_value_in_df = df[region].min() # !! added
    #max_value_in_df = df[region].max() # !! added

    for _, row in filtered_dist_df.iterrows():
        r_value = row['r']
        rr_value = row['rr']
        raw_distance = row['length_miles']
        norm_distance = (raw_distance - min_dist) / (max_dist - min_dist)
        #p_value = 1 - slope * norm_distance
        p_value = 1 * np.exp(-coeff * norm_distance)
        #p_value = norm_distance # placeholder
        # Scale norm_distance using the range of values in df
        #norm_distance_scaled = norm_distance * (max_value_in_df - min_value_in_df) + min_value_in_df # !! added
        # Apply the exponential function with the scaled distance
        #p_value = np.exp(-coeff * norm_distance_scaled)  # !! added


        # Clip pacific_value to ensure it stays between 0 and 1
        p_value = max(0, min(p_value, 1))  ## !! 

        # Find the row in df_weights_updated where the 'r' value matches rr_value
        # if p_value is 1 then keep it at 1 though.
        if df_weights_updated.loc[df_weights_updated['r'] == rr_value, region].values[0] != 1: 
            df_weights_updated.loc[df_weights_updated['r'] == rr_value, region] = p_value


### Plot to check the new weights file
merged_gdf = gdf.merge(df_weights_updated, left_on='rb', right_on='r', how='left')

# Plot the weights
#region_name = 'South_Atlantic' # Change to 'Pacific', 'Mountain', etc. 
fig, ax = plt.subplots()
merged_gdf.set_geometry('geom_of_p_regions').plot(column=region_name, ax=ax, legend=True)
# Add census boundaries
census_without_country.plot(ax=ax, color='red')
plt.title(f'{region_name}, generated weights file')



### Remove the weights that cross the western interconnect.
# These are the p regions in the western interconnect:   p_regions_in_western_interconnect
regions_west_of_interconnect = ['Pacific', 'Mountain']
# Mask for products NOT in the western interconnect list
mask_not_in_western = ~df_weights_updated['r'].isin(p_regions_in_western_interconnect)
# Set 'Pacific' and 'Mountain' to 0 for those p regions outside the western interconnect
df_weights_updated.loc[mask_not_in_western, ['Pacific', 'Mountain']] = 0
# For regions east of the interconnect, set the values for those p regions that are outside to 0. 
df_weights_updated.loc[~mask_not_in_western, df.keys().drop(['r', 'MEX','Pacific','Mountain']).tolist()] = 0
#df_weight_updated.loc[]



df_donna_minus_generated_weights = df_weights_updated.copy()
df_donna_minus_generated_weights.iloc[:,1:] = df.iloc[:,1:-1] - df_weights_updated.iloc[:,1:]


## Make a difference plot between Donna's and my generated file... 
merged_gdf = gdf.merge(df_donna_minus_generated_weights, left_on='rb', right_on='r', how='left')



# Plot the difference in the weights
fig, ax = plt.subplots()
merged_gdf.set_geometry('geom_of_p_regions').plot(column=region_name, ax=ax, legend=True)
# Add census boundaries
census_without_country.plot(ax=ax, color='black')
plt.title(f"{region_name}, Original minus Generated Weights File")


print('Original minus Generated max and min: ')
print(f'{df_donna_minus_generated_weights[region_name].max()}, max')
print(f'{df_donna_minus_generated_weights[region_name].min()}, min')
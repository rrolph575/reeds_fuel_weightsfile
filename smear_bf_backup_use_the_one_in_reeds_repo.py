#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reeds
import geopandas as gpd
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib import rcParams



#%% Global settings
decimals = 4
decay_kms = [50, 100, 150, 200]
nrows = len(decay_kms)
scale=7
decay_func = np.exp
threshold_dist = 200
census_boundaries_gdf = reeds.io.get_dfmap()['cendiv']
country_boundaries_gdf = reeds.io.get_dfmap()['country']



#%% Set spatial resolution
resolution = 'county'  # 'coarse_p_region', 'county'

if resolution == 'coarse_p_region':
    dfmap_dict = reeds.io.get_dfmap()  # this is a dict.
    ifile_shp = 'inputs/shapefiles/US_PCA/US_PCA.shp'
    dfmap = gpd.read_file(ifile_shp) 


if resolution == 'county':
    #dfmap = reeds.io.get_countymap() 
    ifile_shp = 'inputs/shapefiles/US_COUNTY_2022/US_COUNTY_2022.shp'
    dfmap = gpd.read_file(ifile_shp)   


if 'cendiv' not in dfmap.keys():
    # Create a new column in dfmap to store the assigned cendiv
    dfmap['cendiv'] = None

    # Iterate through dfmap and assign the correct cendiv
    for idx, row in dfmap.iterrows():
        # Check if the geometry is within any geometry in census_boundaries_gdf
        for cendiv_idx, cendiv_row in census_boundaries_gdf.iterrows():
            if row['geometry'].within(cendiv_row['geometry']):
                dfmap.at[idx, 'cendiv'] = cendiv_idx  # Assign the cendiv (from the index)
                break  # Stop checking once a match is found
    

#%% Add cendiv if not included in input file
hierarchy = reeds.io.get_hierarchy()
dfgroups = census_boundaries_gdf

if isinstance(dfmap, (pd.DataFrame, gpd.GeoDataFrame)): 
    if 'r' in dfmap.columns:
        dfzones = dfmap.copy() 
    elif 'rb' in dfmap.columns:
        dfzones = dfmap.copy()
else:
    if 'r' in dfmap.keys():
        dfzones = dfmap['r'].copy()
    elif 'rb' in dfmap.keys():
        dfzones = dfmap['rb'].copy()


#%% Smear it out
def smear(dfzones, dfgroups, decay_km=50, decay_func=np.exp):
    dfsmeared = dfzones.copy()
    weights = {}
    distances_km_all = {}

    for r, row in dfzones.iterrows():
        ## Get distance from centroid to edge of all other zones
        ## To get edge-of-polygon-to-edge-of-polygon distance, remove .centroid below
        ## To get centroid-to-centroid distance, add .centroid after dfgroups
        distances_km = dfgroups.distance(row.geometry.centroid) / 1000
        ## !! why is there a 0 in distances_km ? Is it becuase the cetnroid is right on the boundary?
        rb_key = row['rb']
        distances_km_all[rb_key] = distances_km
        ## Weight decays with distance from centroid
        weight = decay_func(-distances_km / decay_km)  # this blurs with all the regions
        weights[rb_key] = weight

    weight_df = pd.DataFrame(weights)
    weight_norm = weight_df / weight_df.sum()
    weight_norm = weight_norm.T


    return weight_norm, distances_km_all

#%% Smear
#weight_norm, distance_all = smear(dfzones, dfgroups, decay_km=decay_km, decay_func=decay_func)
# Merge the weights to the geographic coords 
#merged_gdf = dfzones.merge(weight_norm, left_on=dfzones['rb'], right_on=weight_norm.index, how='left')

cmap = plt.cm.Blues
norm = Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
rcParams.update({'font.size': 18})  # Adjust this value to make fonts larger globally

for region_name in census_boundaries_gdf.index.to_list():
#region_name = "Mountain"

    plt.close()
    f,ax = plt.subplots(
        nrows,1, figsize=(scale*2, scale*nrows*0.8), sharex=True, sharey=True,
        gridspec_kw={'wspace':0, 'hspace':0.01}, dpi=300,
    )
    ax[0].annotate(
        'Decay \nlength (exp)', (-0.05, 1.0), xycoords='axes fraction',
        ha='right', va='bottom', weight='bold', fontsize='x-large',
    )
    for row, decay_km in enumerate(decay_kms):
        weight_norm, distance_all = smear(dfzones, dfgroups, decay_km=decay_km, decay_func=decay_func)
        merged_gdf = dfzones.merge(weight_norm, left_on=dfzones['rb'], right_on=weight_norm.index, how='left')
        ## Data
        merged_gdf.plot(ax=ax[row], column=region_name, vmin=0, vmax=1, cmap=cmap)
        ## Formatting
        # Add census boundaries
        census_boundaries_gdf.boundary.plot(ax=ax[row], color='red')
        ax[row].axis('off')
        if row == 0:
            ax[row].set_title(
                'Weight', y=0.95, weight='bold', fontsize='x-large',
            )
            # Add a colorbar to the figure
            cbar_ax = f.add_axes([0.8, 0.7, 0.02, 0.2])  # Position: [left, bottom, width, height]
            ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
            #cbar_ax.set_label('Weight', fontsize='large', weight='bold')
        ax[row].annotate(
            f'{decay_km} km', (-0.05, 0.5), xycoords='axes fraction',
            ha='right', va='center', weight='bold', fontsize='x-large',
        )

    plt.savefig(f'{region_name}_weights_{resolution}.png', bbox_inches='tight')
    plt.show()


"""
regions= ['Pacific', 'Mountain', 'East_South_Central']
for region_name in regions: 
    fig, ax = plt.subplots()
    merged_gdf.plot(column=region_name, ax=ax, legend=True, vmin=0, vmax=1)
    #merged_gdf.plot(column=region_name, ax=ax, legend=True)
    # Add census boundaries
    census_boundaries_gdf.boundary.plot(ax=ax, color='red')
    plt.title(f"{region_name}, Generated weights file from centroid \n to census boundary, normal weighting")
"""












## Commenting the 'adjusted weights method' for now.
'''
# Implement the adjusted weights so that weights inside the census boundary have even more than outside. Method outlined in 'census divison weighting example.xlsx'
distance_all_df = pd.DataFrame(distance_all).T

# Identify the columns to subtract the threshold from (excluding 'rb' and 'centroid')
distance_columns = [col for col in distance_all_df.columns] # if col not in ['rb', 'centroid']]

distance_all_df_miles = distance_all_df.copy()
for col in distance_columns:
    distance_all_df_miles[col] = distance_all_df[col] * 0.621371

subtract_df = distance_all_df_miles.copy()


# Subtract the threshold distance from the selected columns
subtract_df[distance_columns] = subtract_df[distance_columns] - threshold_dist


# Multiply
multiply_df = subtract_df.copy()
multiply_df[distance_columns] = -1 * subtract_df[distance_columns]

# Clip
# Clip negative values in multiply_df and set them to 0
clip_df = multiply_df.copy()
clip_df[distance_columns] = clip_df[distance_columns].clip(lower=0)


# Total value
# sum all of the clipped values and normalize by that
# Create a copy of clip_df to retain all columns
weight_df = clip_df.copy()

# Iterate over each row in clip_df[distance_columns]
for idx, row in clip_df[distance_columns].iterrows():
    row_sum = row.sum()  # Calculate the sum of the row
    if row_sum > threshold_dist:  # Check if the sum exceeds the threshold
        if ((row > 0) & (row <= threshold_dist)).sum() > 1: # check if more than one cendiv within threshold_dist
            weight_df.loc[idx, distance_columns] = row / row_sum
            # Do adjusted weight so that the region wihtin the census division gets even more weight than the ones near the census boundary    
            # Find the column with the maximum weight
            max_weight_col = weight_df.loc[idx, distance_columns].idxmax()
            # Adjust the weight for the column with the maximum value
            weight_df.loc[idx, max_weight_col] = 0.5 + 0.5 * weight_df.loc[idx, max_weight_col]
            # For the nonzero columns 0.5*weight is the adjusted weight
            nonzero_cols = weight_df.loc[idx, distance_columns][weight_df.loc[idx, distance_columns] > 0].index.tolist()
            nonzero_cols.remove(max_weight_col)
            weight_df.loc[idx, nonzero_cols] = 0.5 * weight_df.loc[idx, nonzero_cols]
        else:
            weight_df.loc[idx, distance_columns] = row / row_sum
    if row_sum == threshold_dist:
        col_with_centroid_in_region = row[row==threshold_dist].idxmax()  # Find the column with the least clipped value
        weight_df.loc[idx, col_with_centroid_in_region] = 1


# Plot the weights using adjusted weights method
# Add column to weight_df that shows the region label 
merged_gdf2 = dfzones.merge(weight_df, left_on=dfzones['rb'], right_on=weight_df.index, how='left')

regions= ['Pacific', 'Mountain', 'East_South_Central']
#for region_name in regions: 
for region_name in distance_columns:
    fig, ax = plt.subplots()
    merged_gdf2.plot(column=region_name, ax=ax, legend=True, vmin=0, vmax=1)
    #merged_gdf.plot(column=region_name, ax=ax, legend=True)
    # Add census boundaries
    census_boundaries_gdf.boundary.plot(ax=ax, color='red')
    plt.title(f"{region_name}, Generated weights file using \nadjusted weights method")

'''
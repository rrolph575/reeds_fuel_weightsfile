#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reeds

#%% Global settings
decimals = 4

#%% Create dummy data
dfmap = reeds.io.get_dfmap()
hierarchy = reeds.io.get_hierarchy()
level = 'cendiv'
dfdata = dfmap[level]
dfdata['value'] = np.random.random(len(dfdata)).round(decimals)

dfzones = dfmap['r'].copy()
dfzones['value_original'] = dfzones.index.map(hierarchy[level]).map(dfdata.value)

#%% Smear it out
def smear(dfzones, decay_km=100, decay_func=np.exp):
    ## Create an aggregated zone for each unique data value
    dfgroups = dfzones.dissolve('value_original')
    dfsmeared = dfzones.copy()
    value_weighted = {}
    for r, row in dfzones.iterrows():
        ## Get distance from centroid to edge of all other zones
        ## To get edge-of-polygon-to-edge-of-polygon distance, remove .centroid below
        ## To get centroid-to-centroid distance, add .centroid after dfgroups
        distances_km = dfgroups.distance(row.geometry.centroid) / 1000
        ## Weight decays with distance from centroid
        weight = decay_func(-distances_km / decay_km)
        ## Take weighted average over all zones
        value_weighted[r] = (weight.index * weight.values).values.sum() / weight.values.sum()

    dfsmeared['value_smeared'] = pd.Series(value_weighted)
    return dfsmeared

#%% Take a look
cmap = plt.cm.turbo
decay_func = np.exp
decay_kms = [50, 100, 200, 300]
nrows = len(decay_kms)
scale = 3
columns = ['value_original', 'value_smeared']
show_buffer = False
show_centroid = False
show_value = True

plt.close()
f,ax = plt.subplots(
    nrows, 2, figsize=(scale*2, scale*nrows*0.8), sharex=True, sharey=True,
    gridspec_kw={'wspace':-0.05, 'hspace':-0.5}, dpi=300,
)
ax[0,0].annotate(
    decay_func.__name__, (-0.05, 1.0), xycoords='axes fraction',
    ha='right', va='bottom', weight='bold', fontsize='x-large',
)
for row, decay_km in enumerate(decay_kms):
    dfsmeared = smear(dfzones, decay_km=decay_km, decay_func=decay_func)
    for col, column in enumerate(columns):
        ## Data
        dfsmeared.plot(ax=ax[row,col], column=column, vmin=0, vmax=1, cmap=cmap)
        ## Formatting
        ax[row,col].axis('off')
        dfsmeared.plot(ax=ax[row,col], facecolor='none', edgecolor='k', lw=0.05)
        if show_buffer:
            dfsmeared.centroid.buffer(decay_km*1000).plot(
                ax=ax[row,col], facecolor='none', edgecolor='w', lw=0.2)
        if show_centroid:
            dfsmeared.centroid.plot(
                ax=ax[row,col], lw=0, marker='o', color='k', markersize=1)
        if show_value:
            for i, _row in dfsmeared.iterrows():
                ax[row,col].annotate(
                    f'{_row[column]*100:.0f}',
                    (_row.geometry.centroid.x, _row.geometry.centroid.y),
                    ha='center', va='center', fontsize=4, color='w',
                )
        dfdata.plot(ax=ax[row,col], facecolor='none', edgecolor='k', lw=0.5)
        if row == 0:
            ax[row,col].set_title(
                column.split('_')[1], y=0.95, weight='bold', fontsize='x-large',
            )
        if col == 0:
            ax[row,col].annotate(
                f'{decay_km} km', (-0.05, 0.5), xycoords='axes fraction',
                ha='right', va='center', weight='bold', fontsize='x-large',
            )
plt.show()

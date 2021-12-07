# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:23:56 2021

@author: strongce
"""
#%% FILE PROCESSING
# Load Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# copy paste the working directory here
path = 'C:\LC_Pro\GCaMP\oud model\damgo\Cal6_10-31-21 plate'
os.chdir(path) # change the working director to new path

id = 'K8' # make this the name of the folder the data is in
spheroid = 'VTA-like (DAMGO WD)' # indicate the group; this will be used as plot titles

# Create a second path that changes the working directory to within the folder
path2 = path+ '\\' +id
os.chdir(path2)

# load the file and convert to a pd df
Results_df = pd.read_csv(path2 + '\\' + 'ROI normalized.csv',
                         sep=",", engine='c', header=0)

# normalize data so that 1 is the lowest value
# first make the time column it's own df
time = Results_df[['time(s)']]
# transpose the Results df and remove the row with time values
Results_wide = (Results_df.transpose()).loc['Roi1':]
# put back in long format
Results_long = Results_wide.transpose()

# create a function that makes the minimum number of every ROI equal 1
def min_1(x):
    return (x+(1-x.min()))

# normalize the data using the min_1 function
Normalized_ROI = Results_long.apply(min_1)
# make a list including the time row and ROI rows
dfs = [time, Normalized_ROI]
# merge the time and ROI rows together
merged = pd.concat(dfs, join = 'outer', axis = 1)


#%% PLOTS -- ALL ROIS

# Line plot averaging ROI with variability (CI) represented with shading
merged_long_format = pd.melt(merged, id_vars=['time(s)'], var_name= 'ROI', 
                              value_name = 'F/F0')

f, ax = plt.subplots(figsize = (10, 5))
ax = sns.lineplot(x = 'time(s)', y='F/F0', color = '.1', ci = 95, linewidth = 3, data = merged_long_format)
sns.set_style('white')
plt.axhline(y=1, color = 'black', linestyle = '--') 
sns.set_style('ticks', {'xtick.minor.size' : 8, 'ytick.minor.size': 8})
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylim(0.75,2)
plt.xlim(0,300)
plt.xlabel("Time (sec)", fontsize = 20)
plt.ylabel("F/F0", fontsize = 20)
plt.title(spheroid, fontsize = 25)
f.savefig("total_ROI_mean_CI.tiff")


# Heat map showing F/F0 values for every ROI measured in LC Pro
Normalized_ROI_wide = Normalized_ROI.transpose()

plt.subplots(figsize=(10,5))
heatmap = sns.heatmap(Normalized_ROI_wide, yticklabels=5, xticklabels=50, vmin=0.9, vmax=2.8,
                      cmap = 'viridis',cbar_kws={'label': 'F/F0'})
plt.xlabel('Frame', fontsize=20)
plt.ylabel('Regions of interest', fontsize=20, labelpad=15)
heatmap.tick_params(labelleft=False)
plt.savefig("total_ROI_heatmap.tiff")


# Correlation matrix for every ROI measured in LC Pro
corr_matrix = Normalized_ROI.corr()

plt.subplots(figsize=(10,5))
corr = sns.heatmap(corr_matrix, yticklabels = 5, xticklabels=5, cmap= 'rocket', annot=False, vmin=-1, vmax=1)
plt.ylabel('Regions of interest', fontsize=20, labelpad=15)
plt.xlabel('Regions of interest', fontsize=20, labelpad=15)
corr.tick_params(labelleft=False, labelbottom=False)
plt.savefig("total_ROI_Correlation_Matrix.tiff")


# Correlation score for total ROIs
total_corr_score = (corr_matrix.sum()/len(corr_matrix)).transpose()
total_corr_score = (total_corr_score.sum()/len(corr_matrix))


#%% RANDOM GENERATION OF 12 ROIs FOR FUTURE PEAK DETECTION ANALYSIS

# Randomly select n number of columns with .sample
random_rois = Normalized_ROI.sample(12, axis=1, random_state=6)
# convert columns into individual np arrays; transpose with .T
wide_arr = (random_rois.values).T

roi1 = wide_arr[0] ; roi2 = wide_arr[1] ; roi3 = wide_arr[2]
roi4 = wide_arr[3] ; roi5 = wide_arr[4] ; roi6 = wide_arr[5]
roi7 = wide_arr[6] ; roi8 = wide_arr[7] ; roi9 = wide_arr[8]
roi10 = wide_arr[9] ; roi11 = wide_arr[11] ; roi12 = wide_arr[11]

# Peak parameters
h = 1.15 ; p = 0.05 ; w = 8

rois = [roi1, roi2, roi3, roi4, roi5, roi6, roi7, roi8, roi9, roi10, roi11, roi12]
dfs = []

for i in rois:
    peaks, _ = find_peaks(i, height=h, prominence = p, width = w)
    peak_freq = np.count_nonzero(peaks)
    ROI = find_peaks(i, height=h, prominence = p, width = w)
    
    plt.plot(i)
    plt.plot(peaks, i[peaks],"D", color = "red")
    plt.axhline(y=1, color = 'black', linestyle = '--')
    plt.xlabel("Time (sec)")
    plt.ylabel("F/F0")
    plt.xlim(0,480)
    plt.show()
    
    for data in ROI:
        df = pd.DataFrame.from_dict(data)
        df.loc['mean'] = df.mean()
        df.insert(0, 'Well_ID', id)
        df.insert(1, 'Peak_Count', peak_freq)
        
        dfs.append(df)
        
del dfs[0::2] # removes the elements from the list that showed peak position
        

rois_merged = pd.concat(dfs, ignore_index=False) # concatenate list of dataframes
rois_merged.to_csv("Detailed_ROI_Data.csv")

rois_merged_wide = rois_merged.transpose()
roi_means_wide = rois_merged_wide[['mean']]
roi_means = roi_means_wide.transpose()


os.chdir(path)
file = id + '.csv'
roi_means.to_csv(path + '\\' + file)



#%% PLOTS -- RANDOM ROIS

# make a list of time and random rois and merge together
dfs2 = [time, random_rois]
merge2 = pd.concat(dfs2, join='outer', axis=1)
# convert to long format for plots
merge2_long = pd.melt(merge2, id_vars=['time(s)'], var_name= 'ROI', 
                              value_name = 'F/F0')

# Line plot showing the individual ROIs overlapping 
sns_plot = sns.relplot(x="time(s)", y="F/F0", hue = "ROI", kind="line", data=merge2_long)
sns_plot.fig.set_size_inches(20,10)
plt.axhline(y=1, color = 'black', linestyle = '--') 
plt.xlabel("Time (sec)", fontsize = 40)
plt.ylabel("F/F0", fontsize = 40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.ylim(0.75,3.5)
plt.xlim(0,300)
sns_plot.savefig("random_ROIs_overlapping.tiff")

# Line plots trellised by individual ROIs
trellis_sns_plot = sns.relplot(x="time(s)", y="F/F0", col = "ROI", col_wrap=4, 
                               kind="line", color = '.3', data=merge2_long)
trellis_sns_plot.savefig("random_ROIs_indiv.tiff")

# Line plot averaging ROI with variability (CI) represented with shading
f, ax = plt.subplots(figsize = (10, 5))
ax = sns.lineplot(x = 'time(s)', y='F/F0', color = '.1', ci = 95, linewidth = 3, data = merge2_long)
plt.axhline(y=1, color = 'black', linestyle = '--') 
sns.set_style('ticks', {'xtick.minor.size' : 8, 'ytick.minor.size': 8})
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylim(0.75, 2.5)
plt.xlim(0,300)
plt.xlabel("Time (sec)", fontsize = 20)
plt.ylabel("F/F0", fontsize = 20)
plt.title(spheroid, fontsize = 25)
f.savefig("random_ROI_mean_CI.tiff")

# Heat map showing F/F0 for the 12 randomly selected ROIs
random_rois_wide = random_rois.transpose()

grid_kws = {"height_ratios":(.5,0.05), "hspace": .3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax = sns.heatmap(random_rois_wide, yticklabels=1, xticklabels=50, cmap = 'viridis', 
                 ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
f.savefig("random_ROI_heatmap.tiff")

# Correlation matrix for the 12 randomly selected ROIs
random_corr_matrix = random_rois.corr()

grid_kws = {"height_ratios":(.8,0.05), "hspace": 0.15} # defines cbar size and distance from matrix
a, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize =(8,8))

ax = sns.heatmap(random_corr_matrix, yticklabels = 1, xticklabels=1, cmap= 'rocket', annot=False, 
                 vmin=-1, vmax=1, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
a.savefig("random_ROI_Correlation_Matrix.tiff")

# Correlation matrix for the 12 randomly selected ROIs with annotations
grid_kws = {"height_ratios":(.8,0.05), "hspace": 0.15} # defines cbar size and distance from matrix
a, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize =(8,8))

ax = sns.heatmap(random_corr_matrix, yticklabels = 1, xticklabels=1, cmap= 'rocket', annot=True, 
                 vmin=-1, vmax=1, ax=ax, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
a.savefig("annotated_ROI_Correlation_Matrix.tiff")

# Correlation score for the 12 randomly selected ROIs
corr_score = (random_corr_matrix.sum()/12)
corr_score_wide = corr_score.transpose()

random_corr_score = corr_score_wide.sum()/12


print('this is your total synchrony score:', total_corr_score)
print('this is your random synchrony score:', random_corr_score)









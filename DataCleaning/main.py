""" The title of the project is Optimising Livestock Production

My first step in this project is to identify signs of successful/unsuccessful
I am using this file to help decipher how I might categorise the various bulls
into groups including High Fertility

To complete this task I will need to do a couple of things
1. Clean & Format Data
2. Convert Data Types and deal with missing data
3. Single Value Plots on Numerical Variables

Many of the method applied have been learned from https://github.com/WillKoehrsen/machine-learning-project-walkthrough/blob/master/Machine%20Learning%20Project%20Part%201.ipynb
"""
# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MissingValues

#Read and display data in a dataframei
dtfm=pd.read_excel('initial_data.xlsx', sheet_name='BD_Research_Fapesp_final',       header=1,usecols=[4,5,32,33,34])

#print out data head
print('\nInfo:')
print(dtfm.info())

#print out the head
print('\nHead')
print(dtfm.head())

#convert output rows from objects to numbers
dtfm = dtfm.replace('.', np.nan)

# Subset of column titles that should be numeric
numeric_cols = ['CLIV','BLAST_D8','CELLS_COUNT']

#Iterate through Columns
for c in list(dtfm.columns):
    if (c in numeric_cols):
        dtfm[c]=dtfm[c].astype(float)

# statistics for each column
print('\nStatistics for each column')
print(dtfm.describe())

# print out a table of colmns and their missing values
print("\nMissing Value Table:")
print(MissingValues.missing_values_table(dtfm))

"""
- Looking at the Missing Value Table makes me thing that the Cell Count not be worth considering. In order to finalise this I need to be able to reason such information statistically and theoretically from other papers

Statistics for each column
             CLIV    BLAST_D8  CELLS_COUNT
count  315.000000  315.000000   182.000000
mean    71.641566   21.375824   169.472456
std     10.413697   11.135731    45.469613
min      0.124075    0.000000     0.250567
25%     65.039683   11.940299   143.000000
50%     72.070374   20.270270   169.750000
75%     79.449472   29.330944   195.000000
max     90.140845   53.623188   269.000000

Missing Value Table:
Your selected dataframe has 5 columns.
There are 4 columns that have missing values.
             Missing Values  % of Total Values
CELLS_COUNT             135               42.6 -> High level of missing data
ANIMAL                    3                0.9
CLIV                      2                0.6
BLAST_D8                  2                0.6

"""

# Probability of the CLIV BLAST_D8 and CELLS_COUNT should all be Gaussian
fig = plt.figure()#Create fig
fig.tight_layout()

#plot style
plt.style.use('seaborn-pastel')

#subplots
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)

#create histograms
ax1.hist(dtfm['CLIV'].dropna(), bins=100)
ax2.hist(dtfm['BLAST_D8'].dropna(), bins=100)
ax3.hist(dtfm['CELLS_COUNT'].dropna(), bins=100)

#x-labels
ax1.title.set_text('Cleavage Rate D3')
ax2.title.set_text('Blastocyst Rate D8')
ax3.title.set_text('Cells Count D3')

#save fig
plt.savefig('HistOfDepVars')


"""
There is nothing that spikes concern about the output variables they all seem to follow a Gaussian distribution going on visual inspection.

Next I will look to see if there exists a correlation between the dependent variables as this may allow me to only focus on 1 or two of them rather than all three.

I will attempt to plot correlations between dependents using a Pearson Correlation Coefficient"""


# CLIV correlations checker
corr_cliv = dtfm.corr()['CLIV'].sort_values()
print('CLIV correlations:')
print(corr_cliv)

# BLAST correlations checker
corr_cliv = dtfm.corr()['BLAST_D8'].sort_values()
print('BLAST_D8 correlations:')
print(corr_cliv)

# CELL COUNT correlations checker
corr_cliv = dtfm.corr()['CELLS_COUNT'].sort_values()
print('CELL_COUNT correlations:')
print(corr_cliv)

"""
Now that the tedious — but necessary — step of data cleaning is complete, we can move on to exploring our data! Exploratory Data Analysis (EDA) is an open-ended process where we calculate statistics and make figures to find trends, anomalies, patterns, or relationships within the data.
In short, the goal of EDA is to learn what our data can tell us. It generally starts out with a high level overview, then narrows in to specific areas as we find interesting parts of the data. The findings may be interesting in their own right, or they can be used to inform our modeling choices, such as by helping us decide which features to use.

"""
# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MissingValues import missing_values_table
from random import seed
from random import randint
import seaborn as sns

#Read and display data in a dataframei
dtfm=pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1')

"""
Single Variable Plots
"""

#Histogram of the BLAST_D8
f, (ax1,ax2,ax3)=plt.subplots(3,1, sharey=True)
plt.style.use('fivethirtyeight')
ax1.hist(dtfm['BLAST_D8'].dropna(), bins=100, edgecolor='k')
ax2.hist(dtfm['CLIV'].dropna(), bins=100, edgecolor='k')
ax3.hist(dtfm['CELLS_COUNT'].dropna(), bins=100, edgecolor='k')
ax1.set_xlabel("BLAST_D8"); ax2.set_xlabel("CLIV");ax3.set_xlabel("CELLS_COUNT");
ax1.set_ylabel("N of Results"); ax2.set_ylabel("N of Results"); ax3.set_ylabel("N of Results");

f.tight_layout()

f.savefig("Hist of Dependent Vars")

"""
Pearson Correlation Coefficient

While the correlation coefficient cannot capture non-linear relationships, it is a good way to start figuring out how variables are related. In Pandas, we can easily calculate the correlations between any columns in a dataframe:
"""

#Create Data frame with dependents and Cell_count
dtfm_CELLS_COUNT=dtfm.drop(["ORDEM", "DATA", "AMOSTRA","REPLICATA","ANIMAL","PARTIDA","CLIV","BLAST_D8"], axis=1)

#Create data frame with dependents and CLIV
dtfm_CLIV=dtfm.drop(["ORDEM", "DATA", "AMOSTRA","REPLICATA","ANIMAL","PARTIDA",  "CELLS_COUNT","BLAST_D8"], axis=1)

#create data with dependents and blast BLAST_D8
dtfm_BLAST_D8=dtfm.drop(["ORDEM", "DATA", "AMOSTRA","REPLICATA","ANIMAL","PARTIDA",  "CLIV","CELLS_COUNT"], axis=1)

# Find all correlations with the score and sort
CLIV_correlations_data = dtfm_CLIV.corr()['CLIV'].sort_values()
BLAST_D8_correlations_data = dtfm_BLAST_D8.corr()['BLAST_D8'].sort_values()
CELLS_COUNT_correlations_data = dtfm_CELLS_COUNT.corr()['CELLS_COUNT'].sort_values()

print("BLAST Correlations Data Top 5 \n{}".format(BLAST_D8_correlations_data.head(5), '\n'))
print("BLAST Correlations Data Bottom \n{}".format(BLAST_D8_correlations_data.tail(5), '\n'))

print("Clivage Correlations Data Top \n{}".format(CLIV_correlations_data.head(5)))
print("Clivage Correlations Data Bottom \n{}".format(CLIV_correlations_data.tail(5)))


print("CELLS COUNT Correlations Data Top \n{}".format(CELLS_COUNT_correlations_data.head(5)))
print("CELLS COUNT Correlations Data Bottom \n{}".format(CELLS_COUNT_correlations_data.tail(5)))

"""
BLAST Correlations Data Top 5
SUB_3_LS      -0.147168
CONC_CAMARA   -0.133755
ALTO          -0.130336
VCL           -0.114562
ALH           -0.098068
Name: BLAST_D8, dtype: float64
BLAST Correlations Data Bottom
MOT_PRE     0.116651
MOT_POS     0.130621
FRAG_CRO    0.149363
SUB_1_RP    0.158840
BLAST_D8    1.000000
Name: BLAST_D8, dtype: float64
Clivage Correlations Data Top
CONC_CAMARA   -0.193136
AD            -0.157065
VF            -0.154023
PI            -0.152501
AI            -0.134380
Name: CLIV, dtype: float64
Clivage Correlations Data Bottom
STATIC_PCT    0.046255
SUB_1_RP      0.080945
FRAG_CRO      0.101268
SUB_4_LP      0.103040
CLIV          1.000000
Name: CLIV, dtype: float64
CELLS COUNT Correlations Data Top
MOT_PRE           -0.184834
MOT_POS           -0.184350
PROGRESSIVE_PCT   -0.138507
RAPID_PCT         -0.125659
BCF               -0.122982
Name: CELLS_COUNT, dtype: float64
CELLS COUNT Correlations Data Bottom
CONC_CAMARA    0.072991
MEDIUM_PCT     0.100983
ALH            0.204568
SLOW_PCT       0.305310

As can be seen there are no particularilly strong -ive or +ive linear correlations between the independent and dependent vars.

To show this in more detail I will put it into a heatmap

"""

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    fig.tight_layout()
    plt.show();


correlation_heatmap(dtfm)






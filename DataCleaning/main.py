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

"""***Aside done after finding the outliers***"""
# Finding Outliers
first_q_cliv = dtfm['CLIV'].describe()['25%']
third_q_cliv = dtfm['CLIV'].describe()['75%']
q_range_cliv=third_q_cliv-first_q_cliv

#remove CLIV outliers are
dtfm=dtfm[(dtfm['CLIV'] > (first_q_cliv - 3 * q_range_cliv)) & (dtfm['CLIV'] <           (third_q_cliv + 3 * q_range_cliv))]
"""***Aside done after finding the outliers***"""

# statistics for each column after removal of outliers
print('\nStatistics for each column after outlier removal')
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

Statistics for each column after outlier removal
             CLIV    BLAST_D8  CELLS_COUNT
count  313.000000  313.000000   180.000000
mean    72.070374   21.475320   171.115891
std      8.942164   11.093061    42.876076
min     49.350649    0.000000    57.000000
25%     65.079365   12.121212   144.875000
50%     72.151899   20.312500   169.875000
75%     79.487179   29.629630   195.437500
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
print('\nCLIV correlations:')
print(corr_cliv)

# BLAST correlations checker
corr_cliv = dtfm.corr()['BLAST_D8'].sort_values()
print('\nBLAST_D8 correlations:')
print(corr_cliv)

# CELL COUNT correlations checker
corr_cliv = dtfm.corr()['CELLS_COUNT'].sort_values()
print('\nCELL_COUNT correlations:')
print(corr_cliv)

"""
Using this Tutorial to help with this section

https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166"""

# import a visualisation library
import seaborn as sns
sns_plot=sns.pairplot(dtfm[['BLAST_D8', 'CELLS_COUNT', 'CLIV']].dropna())
sns_plot.savefig('DependentPairPlot')

"""
There exists a correlation between CLIV and BLAST

CLIV correlations:
CELLS_COUNT    0.164713
BLAST_D8       0.582693
CLIV           1.000000

We will look to focus our analysis on CLIV and BLAST at first for the following reasons

- Literature Points towards these being the most important factors
    - Cleavage, embryo development and blastocyst rates were recorded and compared between Higher and Lower of respective trait groups. Surprisingly, evaluation of isolated effects revealed that lower levels of MB, AI and MP resulted in higher embryo development and blastocyst rates (p<0.05), which was not observed on cleavage rate. We conclude that sperm traits strongly influence embryo development after in vitro fer- tilization (IVF), affecting the zygote competence to achieve blastocyst stage. (Sperm traits on IVP)
    - Oocyte cleavage rate following
47 insemination with sperm from high fertility Holstein Friesian bulls was significantly
48 higher than with sperm from low fertility Holstein Friesian bulls [76.7% (95%CI 60.9
49 to 89.4) and 55.3 (95%CI 40.4 to 69.7) respectively, P = 0.04]. There was no
50 significant effect of bull fertility on blastocyst rate [34.7% (95%CI 21.1 to 49.6) and
51 24.2 % (95%CI 14.1 to 36.0) for the high and low fertility Holstein Friesian bulls,
52 respectively; P = 0.2]. In conclusion, sperm from high fertility bulls tended to be more
53 effective in penetrating artificial mucus and to have an increased ability to fertilise
54 oocytes in vitro; however, once fertilisation occurred subsequent embryo
55 development was not significantly affected by fertility status. (In Vitro Assesment ...)
    - Blastocyst rate was higher in the HF group (29.4%) than in the LF (16.0% - P<0.0001), similarly to embryo development rate (HF = 34.0%; HL = 189%; P<0.0001). There was no significant difference in cleavage rate (HF=86.7%; LF= 84.9%; P= 0.2581), neither in embryo kinetics, in all of the evaluated periods (P>0.05).... In conclusion, early embryo kinetics could not explain the difference in blastocyst rate between high fertility (HF) and low fertility (LF) bulls. Nevertheless, HF bulls had
      more normal fertilization than LF bulls.(Fertilization rate and developmental kinetics of bovine embryos produced using semen from high and low in vitro fertility bulls)

- Blastocyte rate and Clevage seems to have mixed reveiws (this may depend on the species) however both of them seem to produce significant results in some areas and show a correlation in our dataset which is a +ive
- Our data shows neglegible correlation between CELL_COUNT and blastocyst rate
- CELL_COUNT is missing 42% of its data


I propose taking two variables BLASTOCYST_RATE and CLEVAGE_RATE and looking to build a Random Forest that produces
    - Continuous expectancy over CLEVAGE/BLAS
    - Classification of bull into HF or LF - HF CLEV(>76.6) BLAS(29.4) - LF CLEV(69.7 => Upper 95% or 65.04% => Med or 55.3 => Literature) BLAS(16% = Literature, 11% = analysis)
    - Classification of Embryo in 1234 - Dead, Destroy, Transportable, Exportable


    Questions
        - What is Blastocyst Rate? Does it account for all Embryos that are Transportable/Exportable? Is it the %age of Embryo's that reach Blastocyst
        - Do the following bounds seem reasonable for you
            - Classification of bull into HF or LF - HF CLEV(>76.6) BLAS(29.4) - LF CLEV(65.04% => Med or 55.3 => Literature) BLAS(16% = Literature, 12% =        analysis)
"""


# Finding Outliers
first_q_cliv = dtfm['CLIV'].describe()['25%']
third_q_cliv = dtfm['CLIV'].describe()['75%']
q_range_cliv=third_q_cliv-first_q_cliv

#print what the CLIV outliers are
outliers_cliv=dtfm[(dtfm['CLIV'] < (first_q_cliv - 3 * q_range_cliv)) | (dtfm['CLIV'] > (third_q_cliv + 3 * q_range_cliv))]
print("\nOutliers CLIV Head:")
print(outliers_cliv)

 # Finding Outliers
first_q_blast = dtfm['BLAST_D8'].describe()['25%']
third_q_blast = dtfm['BLAST_D8'].describe()['75%']
q_range_blast=third_q_blast-first_q_blast

#print what the BLAST outliers are
outliers_blast=dtfm[(dtfm['BLAST_D8'] < (first_q_blast - 3 * q_range_blast)) | (dtfm['BLAST_D8'] > (third_q_blast + 3 * q_range_blast))]
print("\nOutliers BLAST Head:")
print(outliers_blast)


"""
Outlier consideration

Outliers CLIV Head:
    ANIMAL     BATCH      CLIV   BLAST_D8  CELLS_COUNT
315    NaN   82308.6  8.942164  11.093061    42.876076
316    NaN  0.556996  0.124075   0.516549     0.250567
Outliers BLAST Head:
Empty DataFrame
Columns: [ANIMAL, BATCH, CLIV, BLAST_D8, CELLS_COUNT]
Index: []


Should results with marked as CLIV outliers i.e with Rates 8.94216 and 0.124075 be considered outliers that can be removed from the dataset? This is not needed for main AL implementation but is for EDA?

After removal Pearson Coefficient has incrased from 0.58 - 0.61 which is +ive but maybe not realistic what I want to know is if it is realistic?
"""


"""In conclusion

- Should map off BLAS and CLIV
- CLIV may contain outlier needs addressing
- Correlation exists between BLAST and CLIV
-

"""

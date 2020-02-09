"""
This file will be used for feature engineering

Including:

    Taking our data and creating new features through
    variable transformations, natural log, sqrt, one hot encoding
    categorical
    Feature selection i.e generalising features so we only maintain
    the most relevant featuresin the data

"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_heatmap(dtfm):
    """
    This function can be used to calculate colinearity between features given a dtfm
    The function is then able to plot a .

    USAGE: ccf(dtfm)

    INPUTS:
        dtfm: pandas dataframe

    OUTPUT:
        dtfm

    """
    # calculate correlation matrix
    corr = dtfm.corr()
    # seaborn heatmap
    ax = sns.heatmap(corr, center=0, square=True, linewidths=.5,
            cbar_kws={"shrink": .5}, cbar=True,  cmap='coolwarm',
            vmin=-1, vmax=1)
    sns.despine()
    plt.tight_layout()
    plt.show()

def feature_engineering():
    """
    This FE function can be used to convert discreet categorical data such as
    sup population data which can be measured in the lab into dicreet numerical data.

    i.e if the study were extended to include new bull types such as holstein or
    fresian this would obviously be relative information, feature engineering could
    then be used to encode holstein or fresian as 1 or 0 respectively.

    USAGE:



    NOTE: This feature engineering function could be much imporved with
    greater domain specific knowledge than that of the current practisioner.
    Random Forest algorithms are invariant to monotone transformations
    are not effective at gaining new insights. However, with greater domain knowledge
    it may be possible to multiply or take non-linear calculations by various traits,
    in doing so new useful features could be created."""

if __name__ == "__main__":
    # read in dataframe
    dtfm = pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1', index_col=0)
    #cols to drop
    discrete_cols = ['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA']
    discrete_cols = discrete_cols + ['SUB_1_RP','SUB_2_H','SUB_3_LS','SUB_4_LP']

    dtfm = dtfm.drop(columns=discrete_cols)
    correlation_heatmap(dtfm)

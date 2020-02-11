import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_heatmap(dtfm, title='correlation_heatmap.png', save_fig=False):
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

    if save_fig == True:
        plt.savefig(title)
    else:
        plt.show()


if __name__ == "__main__":
    corr_heatmap = True
    # read in dataframe
    dtfm = pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1', index_col=0)
    #cols to drop
    discrete_cols = ['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA']
    discrete_cols = discrete_cols + ['SUB_1_RP','SUB_2_H','SUB_3_LS','SUB_4_LP']

    dtfm = dtfm.drop(columns=discrete_cols)
    if corr_heatmap:
        correlation_heatmap(dtfm)

"""
This file will be used to create pair plots for EDA

I will create pair plots to understand the relationships between the various
output variables.
"""

import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt

def pair_plot_func(dtfm, save_fig=False, title='pair_plot.png'):
    """
    This function can be used to produce pair plots where the
    upper half is a scatter plot, diagonal is a histogram,
    lower half are density estimates.
    The function will then plot a seaborn pair plot.

    USAGE: pair_plot_func(dtfm)

    INPUT:
        dtfm: pandas dataframe
        (optional)
        plot=True: wheather to show the plot or not

    """
    #drop variables that are na
    dtfm_plot = dtfm.dropna()

    # add correlation coefficient to plot
    def corr_co_func(x1, x2, **kwargs):
        r = np.corrcoef(x1, x2)[0][1]
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.2, .8), xycoords=ax.transAxes)

    # use seaborn to create a grid
    grid = sea.PairGrid(data = dtfm_plot, height = 3)
    # create hist on the diag
    grid.map_diag(plt.hist, edgecolor = 'black', color = 'blue')
    # create lower as scatter
    grid.map_upper(plt.scatter, color='blue')
    # create upper as hist
    grid.map_lower(corr_co_func)
    grid.map_lower(sea.kdeplot, cmap = plt.cm.Reds)

    # Set title
    if save_fig == True:
        plt.savefig(title)
    else:
        plt.show()



if __name__ == "__main__":
    pair_plot = True
    # read in dataframe
    dtfm = pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1', index_col=0)
    # create pair plot of output vars
    if pair_plot:
        pair_plot_func(dtfm[['BLAST_D8', 'CLIV', 'CELLS_COUNT']])


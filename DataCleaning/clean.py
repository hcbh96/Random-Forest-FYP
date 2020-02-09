import pandas as pd
import numpy as np


# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def format_dtfm(dtfm, convert_cols=[], nan=['.'],logger=False):
    """
    This function can be used to format data from a dataframe to the correct type.
    Formatting data is important as it allows us to preform machine learning over
    variables such as discreetly labelled data. i.e if I knew the bull type then
    I would want to include this data as an Integer rather than a string.

    USAGE: format_dtfm(dtfm, logger=False)

    INPUTS:
        dtfm: pandas dataframe
        (optional)
        logger=False: wheather to log info to console

    OUTPUT:
        formatted dataframe
    """
    if logger:
        print('Data Info:\n{}'.format(dtfm.info()))
        print('\nData Head:\n{}'.format(dtfm.head()))

    # convert i in nan to np.nan
    for i in nan:
        dtfm = dtfm.replace(i, np.nan)


    # convert objects to numbers
    for col in convert_cols:
        labels = []
        for i, row in dtfm.iterrows():
            if row[col] not in labels:
                labels.append(row[col])
        # iterate through collected labels replacing them with integers
        for i in range(len(labels)):
            # replace item with index
            dtfm[[col]] = dtfm[[col]].replace(labels[i], i)

    if logger:
        print("Converted cols info:\n{}".format(dtfm[convert_cols].info()))
        print("Converted cols head:\n{}".format(dtfm[convert_cols].head()))

    return dtfm


def remove_missing(dtfm, p_missing=50, logger=False):
    """
    This function can be used to remove column with multiple
    missing values from a dataframe

    USAGE: remove_missing(dtfm, p_missing=50, logger=False)

    INPUTS:
        dtfm: pandas dataframe
        (optional)
        p_missing=50: a what percentage of missing values
            should the variable be removed
        logger=False: whether the func should log progress
            to terminal

    OUTPUT:
        dtfm
    """

    m_dtfm = missing_values_table(dtfm)

    if logger:
        print("Missing Values:\n{}".format(m_dtfm))

    #isolate missing cols
    missing_cols = list(m_dtfm[m_dtfm['% of Total Values'] > p_missing].index)

    if logger:
        print('We will remove {} columns'.format(len(missing_cols)))

    # drop the cols
    dtfm = dtfm.drop(columns = list(missing_cols))

    return dtfm

def find_outliers(dtfm, cols, remove=False, out_file='outliers.xlsx', logger=False):
    """
    This function can be used to find and remove outliers from a dataset

    USAGE: remove_outliers(dtfm, cols, out_file='outliers.xlsx', logger=False)

    INPUTS:
        dtfm: pandas dataframe
        cols: colum titles of cols to check
        (optional)
        remove=False
        out_file='outliers.xlsx': file to pip the removed anomalies to
        logger=False: wheather to print progress or not:

    OUTPUT:
        dataframe
    """

    outlier_frames = []
    # remove outliers from emasured values
    for i in cols:
        first_q = dtfm[i].describe()['25%']
        third_q = dtfm[i].describe()['75%']
        q_range = third_q-first_q
        # calc outliers
        outliers=dtfm[(dtfm[i] < (first_q - 3 * q_range)) | (dtfm[i]  > (third_q +3 * q_range))]
        outlier_frames.append(outliers)
        print("Shape: {}, {}".format(outliers.shape[0], logger))
        if logger == True & outliers.shape[0] > 0:
            print('Outliers {}:\n{}'.format(i, outliers))

    # pipe outliers to out file
    outlier_frames = pd.concat(outlier_frames)
    outlier_frames.to_excel(out_file)

    if logger:
        print("All Outiers:\n{}".format(outlier_frames))

    # remove outliers from dataframe
    if remove:
        dtfm = dtfm.drop(outlier_frames.index)

    return dtfm


if __name__ == '__main__':
    format_data = True
    missing_cols = True
    p_missing = 50
    outliers = True
    remove_outliers = False
    write_new = True
    logger = True

    # inputs
    convert_cols=["AMOSTRA", "REPLICATA", "ANIMAL", "PARTIDA"]
    outlier_cols=["AI","PI","ALTO","FRAG_CRO","MOT_PRE","MOT_POS","CONC_CAMARA","VF","AD","VAP", "VSL","VCL","ALH","BCF","STR","LIN","MOTILE_PCT","PROGRESSIVE_PCT","RAPID_PCT","MEDIUM_PCT","SLOW_PCT","STATIC_PCT"]
    dtfm=pd.read_excel('initial_data.xlsx', sheet_name='BD_Research_Fapesp_final', header=1)

    if format_data == True:
        nan = ['.']
        dtfm = format_dtfm(dtfm,nan=nan,convert_cols=convert_cols,logger=logger)

    if missing_cols:
        dtfm = remove_missing(dtfm, p_missing=p_missing, logger=logger)

    if outliers:
        dtfm = find_outliers(dtfm, outlier_cols, logger=logger, remove=remove_outliers)

    if write_new:
        dtfm.to_excel('cleaned_data.xlsx')



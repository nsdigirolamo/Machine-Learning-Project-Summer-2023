import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datpath = 'dat\\vehicles.csv'
outpath_d = 'out\\dirty\\'
outpath_c = 'out\\clean\\'

#removes all columns from dataset that are >40% null
def filter_na(NA_counts_by_col, threshold = .4):
    cols_pass = []
    for column in NA_counts_by_col.keys():
        if NA_counts_by_col[column]/df.shape[0] < threshold:
            cols_pass.append(column)
    return cols_pass

def price_outlier(df):
    Q1 = df['price'].quantile(.25)
    Q3 = df['price'].quantile(.75)
    IQR = Q3 - Q1
    df_filtered = df.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 *@IQR)')
    return df_filtered

def unique_count(df, col, path):
    uniq = df[col].value_counts()
    with open(path + col + 'counts.txt', mode='a', encoding='utf8') as tf:
        tf.write(uniq.to_string(header=True, index=True))

def mk_boxplot(df, col, path):
    fig = plt.figure()
    pricebox = df.boxplot(col)
    fig.savefig(path + col + '.svg', format='svg')



def create_report(path, df, filename):
    txt_reports = ['region', 'model', 'state', 'manufacturer', 'title_status']
    box_plots = ['price', 'year', 'odometer']
    with open(path + filename, mode='a', encoding='utf8') as f:
        f.write('FILE STATISTICS\n\n')
        shape = df.shape
        f.write('number of rows: ' + str(shape[0]) + '\nnumber of columns: ' + str(shape[1]))
        print('Shape')
        f.write("\n\nlabels\n")
        columns = ', '.join(df.columns)
        f.write(columns)
        print("columns")
        f.write("\n\nunique values per column\n")
        nuniq = df.nunique(axis=0)
        f.write(nuniq.to_string(header=True, index=True))
        print("unique values")
        f.write("\n\nDescription for numerical values\n")
        desc = df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
        f.write(desc.to_string(header=True, index=True))
        print("data description")
        f.write("\n\nNulls\n")
        nulls = df.isnull().sum()
        f.write(nulls.to_string(header=True, index=True))
        print('nulls')
    for col in box_plots:
        mk_boxplot(df, col, path)
        print('Plot for '+ col)
    for col in txt_reports:
        unique_count(df, col, path)
        print('txt report for ' + col)
##################
##################
df = pd.read_csv(datpath, encoding='utf8')
create_report(outpath_d, df, 'initial_stats.txt')

#cleanup    
na_vals = df.isna().sum()
df_clean = df[filter_na(na_vals)]
#recast to remove extreme price outliers
df_clean = df_clean[df_clean['price'].between(500, 300000)]
df_filtered = price_outlier(df_clean)
df_filtered = df_filtered[df_filtered['odometer'].between(0, 300000)]
print('dataframe filtered')
df_filtered.to_csv(outpath_c + 'filtered_data.csv', encoding='utf-8', index=False)
print('new dataframe written')
create_report(outpath_c, df_filtered, 'cleaner_stats.txt')

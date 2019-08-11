#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:00:20 2019

@author: msr
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_files(files, columns):
    
    df_dict = {}

    for i, file in enumerate(files):
        df = pd.read_csv(file, names=columns, header=0)
        file_number = file[-5]
        df['source'] = file_number
        df_dict['df'+file_number] = df
        
    return df_dict





def overlap(df1, df2, column):
    vals1 = df1[column]
    vals2 = df2[column]
    
    if any(vals1.isin(vals2)):
        return "Overlaps found for {}".format(column)
    
    else:
        return "There is no overlap"
    
    
def barplot(df, groupby_col, metric, axis):
    
    df_summarised = df.groupby([groupby_col])[metric].sum().sort_values(ascending=False).to_frame()

    df_summarised = df_summarised.head(20)

    plot = sns.barplot(x = df_summarised.index.get_level_values(0), 
                       y=df_summarised[metric],
                       ax=axis)
    
    plot.set_xlabel(groupby_col, fontsize = 20)
    
    plot.set_ylabel(metric, fontsize = 20)
    
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45);
    
    return df_summarised, plot
    
    
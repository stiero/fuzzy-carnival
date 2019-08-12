#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:00:20 2019

@author: msr
"""
import numpy as np
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
    
    

    
def barplot(df, groupby_col, metric, summary_type, axis, **kwargs):
    
    size = 52
    
    if summary_type == "sum":
        df_summarised = df.groupby([groupby_col])[metric].sum().sort_values(ascending=False).to_frame()

    elif summary_type == "avg":
        df_summarised = df.groupby([groupby_col])[metric].mean().sort_values(ascending=False).to_frame()

    df_summarised = df_summarised.head(size)

    plot = sns.barplot(x = df_summarised.index.get_level_values(0), 
                       y=df_summarised[metric],
                       ax=axis)
    
    plot.set_xlabel(groupby_col, fontsize = 20)
    
    plot.set_ylabel(metric, fontsize = 20)
    
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45);
    
    return df_summarised, plot




def week_number_from_start(df, start_year=0, to_add=0):
    
    week_number = []
    
    year_count = start_year - 1
    

    for date in df.date.sort_values().unique():
        
        date = pd.to_datetime(date)
        
        if date.day == 1 and date.month == 1:
            year_count += 1
            
            if year_count > start_year: 
                to_add += 52
                
        week_number.append(df[df['date'] == date]['date'].dt.week.unique()[0] + to_add)
    
    return week_number
            
            


def other_products(df, sku):
    
    bills = df[df['sku'] == sku]['bill'].unique()
  
    skus = df[df['bill'].isin(bills)]['sku'].unique().tolist()
    
    skus.remove(sku)     
    
    return skus
            
      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                
            
        

    
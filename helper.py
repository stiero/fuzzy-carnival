#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:00:20 2019

@author: msr
"""

def overlap(df1, df2, column):
    vals1 = df1[column]
    vals2 = df2[column]
    
    if any(vals1.isin(vals2)):
        return "Overlaps found for {}".format(column)
    
    else:
        return "There is no overlap"
    
    
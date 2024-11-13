
import pandas as pd
import numpy as np
import os 


def trim_ends(df):
    return df[df['time_to_sleep']>0]

def remove_first_epoch(df):
    return df[df['time_to_sleep']>30]

def resample(df):
    return df.sample(1490)

    

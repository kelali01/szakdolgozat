import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import newton
from datetime import datetime
from scipy.optimize import brentq

def sp500_data():

    sp500_raw = yf.download('^GSPC', start='1990-01-01', end='2025-01-01', interval='1mo')
    # Make sure the index is clean (datetime index)
    sp500_raw.index = pd.to_datetime(sp500_raw.index)

    # Resample monthly to quarterly (last observation)
    sp500 = sp500_raw.resample('QE').last()

    # Keep only Close
    sp500 = sp500[['Close']].copy()
    sp500.rename(columns={'Close': 'index'}, inplace=True)

    # Reset index to a clean 'date' column
    sp500 = sp500.reset_index()
    sp500.rename(columns={'Date': 'date'}, inplace=True)
    sp500_final = pd.DataFrame(columns=['date', 'index'])
    sp500_final['date'] = sp500['date']
    sp500_final['index'] = sp500['index']

    return sp500_final


def index_weighted_cashflows(cashflows, index):
    cashflow_dfs = []
    for fund_id in cashflows['FundID'].unique():
        df1 = cashflows[cashflows['FundID'] == fund_id]
        df = df1.merge(index.rename(columns={'date': 'date'}), on='date', how='left')
        df['index_end'] = df['index'].iloc[-1]
        # # Scaling factor
        df['scale_factor'] = df['index_end'] / df['index']
        df['scaled_cashflow'] = df['Cashflow'] * df['scale_factor']
        cashflow_dfs.append(df)
    return pd.concat(cashflow_dfs, ignore_index=True)

# original newton method for xirr
# def xirr(dates, cashflows):  
#     """Compute IRR (XIRR)"""
#     def f(r):
#         return sum([cf / (1 + r)**((d - dates[0]).days / 365.25) for cf, d in zip(cashflows, dates)])
#     return newton(f, 0.1)

#bretq method for xirr
# Brent's method is more robust than Newton's method for finding roots, especially when the function is not well-behaved or has multiple roots.
def xirr(dates, cashflows):
    def f(r):
        return sum([cf / (1 + r)**((d - dates[0]).days / 365.25) for cf, d in zip(cashflows, dates)])
    
    try:
        return brentq(f, -0.99, 10)  # Safe wide bracket
    except ValueError:
        return np.nan

def ln_pme(df):
    #nav todo
    nav_pme = -1 * (df['scaled_cashflow'].sum())
    ln_cf = np.array(df['Cashflow'])
    ln_cf[-1] += nav_pme

    return xirr(df['date'], ln_cf)

def  ks_pme(df):
    #nav todo
    contributions = df[df['Cashflow'] < 0]['scaled_cashflow'].sum() 
    distributions = df[df['Cashflow'] > 0]['scaled_cashflow'].sum() 
    
    return (distributions) / (-contributions)

def pme_plus(df):
    #nav todo
    contributions = df[df['Cashflow'] < 0]['scaled_cashflow'].sum() 
    distributions = df[df['Cashflow'] > 0]['scaled_cashflow'].sum() 
    lambd = (-contributions) / distributions
    df['pme_plus'] = df['Cashflow'].apply(lambda x: x * lambd if x > 0 else x)
    plus_cf = np.array(df['pme_plus']) 
    
    return xirr(df['date'], plus_cf)

def direct_alpha(df):
    #nav todo
    alpha_cf = np.array(df['scaled_cashflow'])
    
    return xirr(df['date'], alpha_cf)
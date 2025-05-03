from data_gen import simulate_private_equity_cashflows
from pme_calc import sp500_data, xirr,  direct_alpha, index_weighted_cashflows
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vintage_weigths(cashflows):
    vintages = cashflows['VintageYear'].unique()
    weights = {}
    for vintage in vintages:
        vintage_cashflows = cashflows[cashflows['VintageYear'] == vintage]
        weights[vintage] = vintage_cashflows['Cashflow'].sum() / cashflows['Cashflow'].sum()
    return weights

def vintage_aggregate(cashflows):
    vintages = cashflows['VintageYear'].unique()
    weights = {}
    for vintage in vintages:
        vintage_cashflows = cashflows[cashflows['VintageYear'] == vintage]
        weights[vintage] = vintage_cashflows['Cashflow'].sum()
    return weights

def vintage_strategy_weigths(cashflows):
    vintages = cashflows['VintageYear'].unique()
    strategies = cashflows['Strategy'].unique()
    weights = {}
    for vintage in vintages:
        vintage_cashflows = cashflows[cashflows['VintageYear'] == vintage]
        vintage_dict = {}
        for strategy in strategies:
            vintage_strategy_cashflows = cashflows[(cashflows['VintageYear'] == vintage) & (cashflows['Strategy'] == strategy)]
            vintage_dict[strategy] = vintage_strategy_cashflows['Cashflow'].sum() / vintage_cashflows['Cashflow'].sum()

        weights[vintage] = vintage_dict
    return weights

def vintage_strategy_weigths_agg(cashflows):
    vintages = cashflows['VintageYear'].unique()
    strategies = cashflows['Strategy'].unique()
    weights = {}
    for vintage in vintages:
        vintage_dict = {}
        for strategy in strategies:
            vintage_strategy_cashflows = cashflows[(cashflows['VintageYear'] == vintage) & (cashflows['Strategy'] == strategy)]
            vintage_dict[strategy] = vintage_strategy_cashflows['Cashflow'].sum()

        weights[vintage] = vintage_dict
    return weights

def vintage_geo_weigths(cashflows):
    vintages = cashflows['VintageYear'].unique()
    geos = cashflows['Geography'].unique()
    weights = {}
    for vintage in vintages:
        vintage_cashflows = cashflows[cashflows['VintageYear'] == vintage]
        vintage_dict = {}
        for geo in geos:
            vintage_geo_cashflows = cashflows[(cashflows['VintageYear'] == vintage) & (cashflows['Geography'] == geo)]
            vintage_dict[geo] = vintage_geo_cashflows['Cashflow'].sum() / vintage_cashflows['Cashflow'].sum()

        weights[vintage] = vintage_dict
    return weights

def vintage_geo_weigths_agg(cashflows):
    vintages = cashflows['VintageYear'].unique()
    geos = cashflows['Geography'].unique()
    weights = {}
    for vintage in vintages:
        vintage_dict = {}
        for geo in geos:
            vintage_geo_cashflows = cashflows[(cashflows['VintageYear'] == vintage) & (cashflows['Geography'] == geo)]
            vintage_dict[geo] = vintage_geo_cashflows['Cashflow'].sum()

        weights[vintage] = vintage_dict
    return weights

def fund_weights(cashflows):
    funds = cashflows['FundID'].unique()
    weights = {}
    for fund in funds:
        fund_cashflows = cashflows[cashflows['FundID'] == fund]
        weights[fund] = -fund_cashflows['Cashflow'].sum()
    return weights

def  ks_pme(df):

    contributions = df[df['scaled_cashflow'] < 0]['scaled_cashflow'].sum() 
    distributions = df[df['scaled_cashflow'] > 0]['scaled_cashflow'].sum() 
    
    return (distributions) / (-contributions)

def model_builder(cashflows, port_cfs):

    # index
    sp500 = sp500_data()
    # Market
    df = index_weighted_cashflows(cashflows, sp500)
    aggregated_df = df.groupby('date', as_index=False)['scaled_cashflow'].sum()
    market_ks_pme = ks_pme(aggregated_df)
    market_da = direct_alpha(aggregated_df)

    # Portfolio
    port_df = index_weighted_cashflows(port_cfs, sp500)
    aggregated_portfolio_df = port_df.groupby('date', as_index=False)['scaled_cashflow'].sum()
    port_ks_pme = ks_pme(aggregated_portfolio_df)
    port_da = direct_alpha(aggregated_portfolio_df)

    cashflows_contr = cashflows[cashflows['Cashflow'] < 0]
    port_cfs_contr = port_cfs[port_cfs['Cashflow'] < 0]
    vintage_market_contr = vintage_weigths(cashflows_contr)
    vintage_port_contr = vintage_weigths(port_cfs_contr)
    vintage_market_contr_agg = vintage_aggregate(cashflows_contr)

    # Timing alpha

    def adjust_cashflow(row):
        vintage = row['VintageYear']
        cf = row['Cashflow']
        # Contribution (negative cashflow)
        weight = -1 * (vintage_port_contr[vintage] / vintage_market_contr_agg[vintage])
        return cf * weight

    cashflows['adjusted_cashflow'] = cashflows.apply(adjust_cashflow, axis=1)
    vintage_cashflows = pd.DataFrame({
        'date': cashflows['date'],
        'FundID': cashflows['FundID'],
        'Cashflow': cashflows['adjusted_cashflow']})

    vintage_df = index_weighted_cashflows(vintage_cashflows, sp500)
    vintage_aggregated_df = vintage_df.groupby('date', as_index=False)['scaled_cashflow'].sum()
    timing_ks_pme = ks_pme(vintage_aggregated_df)
    timing_da = direct_alpha(vintage_aggregated_df)

    # Strategy alpha
    vintage_strategy_port_contr = vintage_strategy_weigths(port_cfs_contr)
    vintage_strategy_market_contr_agg = vintage_strategy_weigths_agg(cashflows_contr)

    def vint_strat_cashflow(row):
        vintage = row['VintageYear']
        strategy = row['Strategy']
        cf = row['Cashflow']
        # Contribution (negative cashflow)
        weight = -1 * (vintage_strategy_port_contr[vintage][strategy] * vintage_market_contr[vintage] / vintage_strategy_market_contr_agg[vintage][strategy])
        
        return cf * weight

    # Apply the function row-wise
    cashflows['vint_strat_adjusted_cashflow'] = cashflows.apply(vint_strat_cashflow, axis=1)
    strategy_cashflows = pd.DataFrame({
        'date': cashflows['date'],
        'FundID': cashflows['FundID'],
        'Cashflow': cashflows['vint_strat_adjusted_cashflow']})

    strategy_df = index_weighted_cashflows(strategy_cashflows, sp500)
    strategy_aggregated_df = strategy_df.groupby('date', as_index=False)['scaled_cashflow'].sum()
    strategy_ks_pme = ks_pme(strategy_aggregated_df)
    strategy_da = direct_alpha(strategy_aggregated_df)

    # Geography alpha
    vintage_geo_port_contr = vintage_geo_weigths(port_cfs_contr)
    vintage_geo_market_contr_agg = vintage_geo_weigths_agg(cashflows_contr)

    def vint_geo_cashflow(row):
        vintage = row['VintageYear']
        geo = row['Geography']
        cf = row['Cashflow']
        # Contribution (negative cashflow)
        weight = -1 * (vintage_geo_port_contr[vintage][geo] * vintage_market_contr[vintage] / vintage_geo_market_contr_agg[vintage][geo])

        return cf * weight

    # Apply the function row-wise
    cashflows['vint_geo_adjusted_cashflow'] = cashflows.apply(vint_geo_cashflow, axis=1)
    geo_cashflows = pd.DataFrame({
        'date': cashflows['date'],
        'FundID': cashflows['FundID'],
        'Cashflow': cashflows['vint_geo_adjusted_cashflow']})
    
    geo_df = index_weighted_cashflows(geo_cashflows, sp500)
    geo_aggregated_df = geo_df.groupby('date', as_index=False)['scaled_cashflow'].sum()
    geo_ks_pme = ks_pme(geo_aggregated_df)
    geo_da = direct_alpha(geo_aggregated_df)


    # Sizing alpha
    eq_port_contr = fund_weights(port_cfs_contr)

    def eq_calc(row):
        fund = row['FundID']
        cf = row['Cashflow']
        
        return cf / eq_port_contr[fund]

    # Apply the function row-wise
    port_cfs['eq_cashflow'] = port_cfs.apply(eq_calc, axis=1)

    eq_ports = pd.DataFrame({
        'date': port_cfs['date'],
        'FundID': port_cfs['FundID'],
        'Cashflow': port_cfs['eq_cashflow']})
    
    eq_df = index_weighted_cashflows(eq_ports, sp500)
    eq_aggregated_df = eq_df.groupby('date', as_index=False)['scaled_cashflow'].sum()

    eq_ks_pme = ks_pme(eq_aggregated_df)
    eq_da = direct_alpha(eq_aggregated_df)


    timing_alpha_ks_pme = timing_ks_pme - market_ks_pme
    timing_alpha_da = timing_da - market_da
    strategy_alpha_ks_pme = strategy_ks_pme - market_ks_pme
    strategy_alpha_da = strategy_da - market_da
    geo_alpha_ks_pme = geo_ks_pme - market_ks_pme
    geo_alpha_da = geo_da - market_da
    sizing_alpha_ks_pme = port_ks_pme - eq_ks_pme
    sizing_alpha_da = port_da - eq_da
    residual_alpha_ks_pme = port_ks_pme - market_ks_pme - (timing_alpha_ks_pme + strategy_alpha_ks_pme + geo_alpha_ks_pme + sizing_alpha_ks_pme)
    residual_alpha_da = port_da - market_da - (timing_alpha_da + strategy_alpha_da + geo_alpha_da + sizing_alpha_da)

    moic_decomposition = {
    'Portfolio KS-PME': port_ks_pme,
    'Market KS-PME': market_ks_pme,
    'Timing Alpha KS-PME': timing_alpha_ks_pme,
    'Strategy Alpha KS-PME': strategy_alpha_ks_pme,
    'Geography Alpha KS-PME': geo_alpha_ks_pme,
    'Sizing Alpha KS-PME': sizing_alpha_ks_pme,
    'Residual Alpha KS-PME': residual_alpha_ks_pme
    }

    # Create a dictionary for IRR components
    irr_decomposition = {
        'Portfolio Direct Alpha': port_da,
        'Market Direct Alpha': market_da,
        'Timing Direct Alpha': timing_alpha_da,
        'Strategy Direct Alpha': strategy_alpha_da,
        'Geography Direct Alpha': geo_alpha_da,
        'Sizing Direct Alpha': sizing_alpha_da,
        'Residual Direct Alpha': residual_alpha_da
    }
    return moic_decomposition, irr_decomposition
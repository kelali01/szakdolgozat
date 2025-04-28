from data_gen import simulate_private_equity_cashflows
from pme_calc import moic, xirr
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


def model_builder(cashflows, port_cfs):
    # Market
    aggregated_df = cashflows.groupby('date', as_index=False)['Cashflow'].sum()
    market_moic = moic(cashflows)
    market_irr = xirr(aggregated_df['date'], aggregated_df['Cashflow'])

    # Portfolio
    aggregated_portfolio_df = port_cfs.groupby('date', as_index=False)['Cashflow'].sum()
    port_moic = moic(port_cfs)
    port_irr = xirr(aggregated_portfolio_df['date'], aggregated_portfolio_df['Cashflow'])

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
    timing_moic = (cashflows[cashflows['adjusted_cashflow'] > 0]['adjusted_cashflow'].sum()) / (-cashflows[cashflows['adjusted_cashflow'] < 0]['adjusted_cashflow'].sum())
    timing_aggregated_df = cashflows.groupby('date', as_index=False)['adjusted_cashflow'].sum()
    timing_irr = xirr(timing_aggregated_df['date'], timing_aggregated_df['adjusted_cashflow'])

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
    strategy_moic = (cashflows[cashflows['vint_strat_adjusted_cashflow'] > 0]['vint_strat_adjusted_cashflow'].sum()) / (-cashflows[cashflows['vint_strat_adjusted_cashflow'] < 0]['vint_strat_adjusted_cashflow'].sum())
    strategy_aggregated_df = cashflows.groupby('date', as_index=False)['vint_strat_adjusted_cashflow'].sum()
    strategy_irr = xirr(strategy_aggregated_df['date'], strategy_aggregated_df['vint_strat_adjusted_cashflow'])

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
    geo_moic = (cashflows[cashflows['vint_geo_adjusted_cashflow'] > 0]['vint_geo_adjusted_cashflow'].sum()) / (-cashflows[cashflows['vint_geo_adjusted_cashflow'] < 0]['vint_geo_adjusted_cashflow'].sum())
    geo_aggregated_df = cashflows.groupby('date', as_index=False)['vint_geo_adjusted_cashflow'].sum()
    geo_irr = xirr(geo_aggregated_df['date'], geo_aggregated_df['vint_geo_adjusted_cashflow'])

    # Sizing alpha
    eq_port_contr = fund_weights(port_cfs_contr)

    def eq_calc(row):
        fund = row['FundID']
        cf = row['Cashflow']
        
        return cf / eq_port_contr[fund]

    # Apply the function row-wise
    port_cfs['eq_cashflow'] = port_cfs.apply(eq_calc, axis=1)
    eq_moic = (port_cfs[port_cfs['eq_cashflow'] > 0]['eq_cashflow'].sum()) / (-port_cfs[port_cfs['eq_cashflow'] < 0]['eq_cashflow'].sum())
    eq_aggregated_df = port_cfs.groupby('date', as_index=False)['eq_cashflow'].sum()
    eq_irr = xirr(eq_aggregated_df['date'], eq_aggregated_df['eq_cashflow'])

    timing_alpha_moic = timing_moic - market_moic
    timing_alpha_irr = timing_irr - market_irr
    strategy_alpha_moic = strategy_moic - market_moic
    strategy_alpha_irr = strategy_irr - market_irr
    geo_alpha_moic = geo_moic - market_moic 
    geo_alpha_irr = geo_irr - market_irr
    sizing_alpha_moic = port_moic - eq_moic
    sizing_alpha_irr = port_irr - eq_irr
    residual_alpha_moic = port_moic - market_moic - (timing_alpha_moic + strategy_alpha_moic + geo_alpha_moic + sizing_alpha_moic)
    residual_alpha_irr = port_irr - market_irr - (timing_alpha_irr + strategy_alpha_irr + geo_alpha_irr + sizing_alpha_irr)

    moic_decomposition = {
    'Portfolio MOIC': port_moic,
    'Market MOIC': market_moic,
    'Timing Alpha MOIC': timing_alpha_moic,
    'Strategy Alpha MOIC': strategy_alpha_moic,
    'Geography Alpha MOIC': geo_alpha_moic,
    'Sizing Alpha MOIC': sizing_alpha_moic,
    'Residual Alpha MOIC': residual_alpha_moic
    }

    # Create a dictionary for IRR components
    irr_decomposition = {
        'Portfolio IRR': port_irr,
        'Market IRR': market_irr,
        'Timing Alpha IRR': timing_alpha_irr,
        'Strategy Alpha IRR': strategy_alpha_irr,
        'Geography Alpha IRR': geo_alpha_irr,
        'Sizing Alpha IRR': sizing_alpha_irr,
        'Residual Alpha IRR': residual_alpha_irr
    }
    return moic_decomposition, irr_decomposition
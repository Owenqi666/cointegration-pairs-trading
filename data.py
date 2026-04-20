import yfinance as yf
import pandas as pd
import numpy as np


def load_pair(ticker_y, ticker_x, start, end):
    # download both tickers in one call
    raw = yf.download([ticker_y, ticker_x], start=start, end=end,
                      auto_adjust=True, progress=False)['Close']

    # flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prices = raw[[ticker_y, ticker_x]].copy()

    # align dates, drop any row with missing data
    prices = prices.dropna()
    return prices


def load_market(ticker, start, end):
    # market benchmark for market-neutrality regression
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()
    return raw.dropna()


def load_rf(start, end):
    # ^IRX is the 13-week T-bill yield, quoted as annualized percent
    irx = yf.download('^IRX', start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    if isinstance(irx, pd.DataFrame):
        irx = irx.squeeze()

    # convert to decimal annual rate
    rf = irx / 100.0
    return rf.dropna()


if __name__ == '__main__':
    # smoke test
    prices = load_pair('GS', 'MS', '2016-01-01', '2026-01-01')
    print('pair shape:', prices.shape)
    print(prices.head())
    print(prices.tail())

    market = load_market('SPY', '2016-01-01', '2026-01-01')
    print('\nmarket shape:', market.shape)
    print('market mean:', float(market.mean()))

    rf = load_rf('2016-01-01', '2026-01-01')
    print('\nrf shape:', rf.shape)
    print('rf mean:', float(rf.mean()))
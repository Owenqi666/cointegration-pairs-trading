import numpy as np
import pandas as pd

np.random.seed(42)

from pairs import get_pair
from data import load_pair, load_market, load_rf
from cointegration import hedge_ratio, adf_test
from metrics import summarize


def rolling_cointegration(prices, window=60):
    # at each day t, fit OLS and run ADF on the trailing window [t-w+1, t]
    # returns a DataFrame with beta_t, pvalue_t, zscore_t aligned to prices.index
    y = prices.iloc[:, 0].values.astype(float)
    x = prices.iloc[:, 1].values.astype(float)
    n = len(y)

    beta_arr = np.full(n, np.nan)
    pvalue_arr = np.full(n, np.nan)
    z_arr = np.full(n, np.nan)

    for t in range(window - 1, n):
        y_win = y[t - window + 1: t + 1]
        x_win = x[t - window + 1: t + 1]

        # step 1: OLS for hedge ratio on the window
        alpha, beta = hedge_ratio(y_win, x_win)
        residuals = y_win - alpha - beta * x_win

        # step 2: ADF on residuals gives current cointegration p-value
        pv = adf_test(residuals)

        # step 3: z-score of the most recent residual, standardized within window
        sigma = residuals.std()
        if sigma > 0:
            z_t = (residuals[-1] - residuals.mean()) / sigma
        else:
            z_t = np.nan

        beta_arr[t] = beta
        pvalue_arr[t] = pv
        z_arr[t] = z_t

    return pd.DataFrame({
        'beta': beta_arr,
        'pvalue': pvalue_arr,
        'zscore': z_arr,
    }, index=prices.index)


def generate_rolling_signal(zscore, pvalue, threshold=0.05,
                            entry=2.0, exit_th=0.5, stop=3.0):
    # same state machine as signal.generate_signal, plus a cointegration gate:
    # when pvalue > threshold the pair is non-tradable and position is forced flat
    z_arr = zscore.values
    p_arr = pvalue.values
    n = len(z_arr)
    signal = np.zeros(n)
    pos = 0

    for i in range(n):
        zi = z_arr[i]
        pi = p_arr[i]

        # gate: not currently cointegrated -> no trading, close any open position
        if np.isnan(pi) or pi > threshold:
            signal[i] = 0
            pos = 0
            continue

        # warmup or singular spread -> flat
        if np.isnan(zi):
            signal[i] = 0
            pos = 0
            continue

        if pos == 0:
            if zi > entry:
                pos = -1
            elif zi < -entry:
                pos = 1
        elif pos == -1:
            if abs(zi) < exit_th or abs(zi) > stop:
                pos = 0
        elif pos == 1:
            if abs(zi) < exit_th or abs(zi) > stop:
                pos = 0

        signal[i] = pos

    return pd.Series(signal, index=zscore.index)


def run_rolling_backtest(prices, signal, beta_series,
                         capital=1.0, fee_rate=0.0005):
    # T+1 execution, beta locked at opening and held constant until close
    effective_signal = signal.shift(1).fillna(0)

    y = prices.iloc[:, 0].values.astype(float)
    x = prices.iloc[:, 1].values.astype(float)
    b = beta_series.values
    sig = effective_signal.values
    n = len(y)

    pnl = np.zeros(n)
    cost = np.zeros(n)
    n_y = 0.0
    n_x = 0.0

    for i in range(1, n):
        prev_sig = sig[i-1]
        curr_sig = sig[i]

        if curr_sig != prev_sig:
            # close old position (n_y, n_x from prior opening)
            if prev_sig != 0:
                cost[i] += fee_rate * (n_y * y[i] + n_x * x[i])
                n_y = 0.0
                n_x = 0.0
            # open new position using today's beta from the rolling fit.
            # invariant: when signal != 0, generate_rolling_signal already
            # guaranteed pvalue is finite, which implies beta is finite too
            # since both are computed in the same window of rolling_cointegration
            if curr_sig != 0:
                beta_trade = b[i]
                denom = y[i] + beta_trade * x[i]
                n_y = capital / denom
                n_x = beta_trade * capital / denom
                cost[i] += fee_rate * (n_y * y[i] + n_x * x[i])

        if curr_sig != 0:
            pnl[i] = curr_sig * (n_y * (y[i] - y[i-1]) - n_x * (x[i] - x[i-1]))

        pnl[i] -= cost[i]

    capital_series = capital + np.cumsum(pnl)
    daily_returns = pnl / capital
    n_trades = int((effective_signal.diff().abs() > 0).sum())

    return {
        'capital': pd.Series(capital_series, index=prices.index),
        'returns': pd.Series(daily_returns, index=prices.index),
        'pnl': pd.Series(pnl, index=prices.index),
        'cost': pd.Series(cost, index=prices.index),
        'n_trades': n_trades,
    }


def run_rolling_strategy(pair, window=60, threshold=0.05,
                         start='2016-01-01', end='2026-01-01'):
    prices = load_pair(pair['ticker_y'], pair['ticker_x'], start, end)
    market = load_market('SPY', start, end)
    rf = load_rf(start, end)

    # stage 1: rolling cointegration estimates beta_t, pvalue_t, zscore_t
    rc = rolling_cointegration(prices, window=window)

    # stage 2: signal with cointegration gate at threshold
    signal = generate_rolling_signal(rc['zscore'], rc['pvalue'],
                                     threshold=threshold)

    # stage 3: backtest with time-varying beta locked at each opening
    bt = run_rolling_backtest(prices, signal, rc['beta'])

    # stage 4: full performance and market-neutrality metrics
    m = summarize(bt['capital'], bt['returns'], rf, market)

    # gate activity: fraction of days where pair was tradable
    is_active = (rc['pvalue'] < threshold) & rc['pvalue'].notna()
    active_days = int(is_active.sum())
    total_days = int(rc['pvalue'].notna().sum())

    return {
        'pair': pair,
        'window': window,
        'threshold': threshold,
        'rolling_coint': rc,
        'signal': signal,
        'capital': bt['capital'],
        'returns': bt['returns'],
        'n_trades': bt['n_trades'],
        'active_days': active_days,
        'total_days': total_days,
        **m,
    }


def print_rolling_report(result):
    pair = result['pair']
    print(f'\n{"="*60}')
    print(f"Rolling Cointegration Strategy: {pair['label']}")
    print(f"  window = {result['window']}d, "
          f"threshold = {result['threshold']}")
    print(f'{"="*60}')

    active_pct = (100.0 * result['active_days'] / result['total_days']
                  if result['total_days'] > 0 else 0.0)
    print('\n=== Cointegration gate ===')
    print(f"tradable days: {result['active_days']} / {result['total_days']} "
          f"({active_pct:.1f}%)")

    total_return = float(result['capital'].iloc[-1]) - 1.0
    print('\n=== Performance ===')
    print(f"total return: {total_return:.4f}")
    print(f"ann return: {result['ann_return']:.4f}")
    print(f"ann vol: {result['ann_vol']:.4f}")
    print(f"sharpe: {result['sharpe']:.4f}")
    print(f"max drawdown: {result['max_drawdown']:.4f}")
    print(f"n_trades: {result['n_trades']}")

    print('\n=== Market neutrality ===')
    print(f"beta_m: {result['beta_m']:.4f}  "
          f"(p={result['beta_pvalue']:.3f})")
    print(f"alpha (daily): {result['alpha_daily']:.6f}  "
          f"(p={result['alpha_pvalue']:.3f})")
    print(f"r_squared: {result['r_squared']:.4f}")


if __name__ == '__main__':
    pair = get_pair('NVDAMSFT')
    result = run_rolling_strategy(pair, window=60, threshold=0.05,
                                  start='2016-01-01', end='2026-01-01')
    print_rolling_report(result)
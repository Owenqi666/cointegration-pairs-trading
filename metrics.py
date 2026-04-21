import numpy as np
import pandas as pd
import statsmodels.api as sm


def annualized_return(capital, days=252, base=None):
    # geometric annualization of total return
    # base: optional explicit initial capital. If None, infers from capital.iloc[0]
    # (which only equals the true initial if the first-day return was 0).
    c_0 = float(capital.iloc[0]) if base is None else float(base)
    total = float(capital.iloc[-1]) / c_0 - 1.0
    t = len(capital)
    if t <= 1:
        return 0.0
    return (1.0 + total) ** (days / t) - 1.0


def annualized_vol(returns, days=252):
    # daily std scaled by sqrt(days)
    return float(returns.std()) * np.sqrt(days)


def sharpe_ratio(returns, rf, days=252):
    # rf is a Series of daily-observed annualized rates; take mean as the
    # period-average risk-free rate to subtract from annualized return
    rf_mean = float(rf.reindex(returns.index).ffill().mean())
    r_ann = float(returns.mean()) * days
    vol_ann = annualized_vol(returns, days)
    if vol_ann == 0:
        return np.nan
    return (r_ann - rf_mean) / vol_ann


def max_drawdown(capital):
    # max peak-to-trough loss as a fraction of the running peak
    running_peak = capital.cummax()
    drawdown = (running_peak - capital) / running_peak
    return float(drawdown.max())


def market_beta(strategy_returns, market_prices, rf):
    # CAPM regression on excess returns: r_s - rf = alpha + beta * (r_m - rf)
    # a pairs trading strategy should have beta ~= 0 if truly market-neutral
    market_returns = market_prices.pct_change().dropna()

    # align all three series on common dates
    df = pd.concat([strategy_returns, market_returns, rf], axis=1).dropna()
    df.columns = ['r_s', 'r_m', 'rf_ann']

    # convert annualized rf to daily decimal
    df['rf_daily'] = df['rf_ann'] / 252.0

    # excess returns
    y = df['r_s'] - df['rf_daily']
    x = df['r_m'] - df['rf_daily']

    x_const = sm.add_constant(x.values)
    model = sm.OLS(y.values, x_const).fit()

    return {
        'alpha_daily': float(model.params[0]),
        'beta_m': float(model.params[1]),
        'alpha_pvalue': float(model.pvalues[0]),
        'beta_pvalue': float(model.pvalues[1]),
        'r_squared': float(model.rsquared),
    }


def summarize(capital, returns, rf, market_prices, days=252, base=None):
    # one-stop metric bundle for a single backtest run
    r_ann = annualized_return(capital, days, base=base)
    vol_ann = annualized_vol(returns, days)
    sharpe = sharpe_ratio(returns, rf, days)
    mdd = max_drawdown(capital)
    mkt = market_beta(returns, market_prices, rf)

    return {
        'ann_return': r_ann,
        'ann_vol': vol_ann,
        'sharpe': sharpe,
        'max_drawdown': mdd,
        'alpha_daily': mkt['alpha_daily'],
        'beta_m': mkt['beta_m'],
        'alpha_pvalue': mkt['alpha_pvalue'],
        'beta_pvalue': mkt['beta_pvalue'],
        'r_squared': mkt['r_squared'],
    }


if __name__ == '__main__':
    # full pipeline smoke test — change pair key here to test another
    from pairs import load_smoke
    from data import load_market, load_rf
    from cointegration import engle_granger, half_life
    from signals import compute_spread, compute_zscore, generate_signal
    from backtest import run_backtest

    prices, y, x, pair = load_smoke('GSMS')
    market = load_market('SPY', '2016-01-01', '2026-01-01')
    rf = load_rf('2016-01-01', '2026-01-01')
    print(f"=== Smoke test: {pair['label']} ===")

    result = engle_granger(y, x)
    beta = result['beta']
    spread = compute_spread(y, x, beta)
    hl = half_life(spread)
    w = int(round(hl)) if not np.isnan(hl) else 30
    z = compute_zscore(spread, w)
    signal = generate_signal(z)
    bt = run_backtest(prices, signal, beta, capital=1.0, fee_rate=0.0005)

    m = summarize(bt['capital'], bt['returns'], rf, market)

    print('\n=== Performance metrics ===')
    print(f"annualized return : {m['ann_return']:.4f}")
    print(f"annualized vol    : {m['ann_vol']:.4f}")
    print(f"sharpe ratio      : {m['sharpe']:.4f}")
    print(f"max drawdown      : {m['max_drawdown']:.4f}")

    print('\n=== Market neutrality (CAPM regression) ===')
    print(f"alpha (daily)     : {m['alpha_daily']:.6f}")
    print(f"alpha p-value     : {m['alpha_pvalue']:.4f}")
    print(f"beta_m            : {m['beta_m']:.4f}")
    print(f"beta_m p-value    : {m['beta_pvalue']:.4f}")
    print(f"r_squared         : {m['r_squared']:.4f}")

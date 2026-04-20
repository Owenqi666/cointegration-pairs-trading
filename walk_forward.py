import numpy as np
import pandas as pd

from data import load_pair, load_market, load_rf
from cointegration import engle_granger, half_life
from signal import compute_spread, compute_zscore, generate_signal
from backtest import run_backtest
from metrics import summarize, market_beta


def get_windows(start_year=2016, end_year=2026, train_years=3):
    # rolling 3-year train / 1-year test, stepped by 1 year (a.k.a. rolling-
    # origin or sliding-window walk-forward; NOT expanding-origin, which would
    # keep the start year fixed and grow the train window each fold)
    windows = []
    for test_year in range(start_year + train_years, end_year):
        train_start = f'{test_year - train_years}-01-01'
        train_end = f'{test_year - 1}-12-31'
        test_start = f'{test_year}-01-01'
        test_end = f'{test_year}-12-31'
        windows.append((train_start, train_end, test_start, test_end))
    return windows


def _skipped(train_s, train_e, test_s, test_e, reason, pvalue=None, hl=None):
    # uniform skip record so aggregation logic can filter on 'status'
    return {
        'train_start': train_s, 'train_end': train_e,
        'test_start': test_s, 'test_end': test_e,
        'status': 'skipped',
        'skip_reason': reason,
        'pvalue': pvalue,
        'half_life': hl,
    }


def run_fold(prices, market, rf, train_s, train_e, test_s, test_e):
    # train-only slice for cointegration estimation (no lookahead)
    prices_train = prices.loc[train_s:train_e]
    prices_test = prices.loc[test_s:test_e]
    prices_full = prices.loc[train_s:test_e]

    if len(prices_train) < 100 or len(prices_test) < 20:
        return _skipped(train_s, train_e, test_s, test_e, 'insufficient data')

    y_train = prices_train.iloc[:, 0]
    x_train = prices_train.iloc[:, 1]

    # train phase: estimate cointegration
    # threshold=0.10 (not the default 0.05): ADF test has limited statistical
    # power on 3-year windows, so 0.10 is the standard in pairs-trading
    # literature (Alexander 2008, Vidyamurthy 2004) to avoid systematic
    # false rejection of weakly-but-genuinely cointegrated pairs
    coint = engle_granger(y_train, x_train, threshold=0.10)
    if not coint['is_coint']:
        return _skipped(train_s, train_e, test_s, test_e,
                        'not cointegrated', pvalue=coint['pvalue'])

    beta = coint['beta']
    hl = half_life(coint['residuals'])
    if np.isnan(hl):
        return _skipped(train_s, train_e, test_s, test_e,
                        'invalid half-life', pvalue=coint['pvalue'])

    w = int(round(hl))

    # test phase: spread and z-score use full (train+test) data for warmup,
    # then signal is sliced to the test period only
    y_full = prices_full.iloc[:, 0]
    x_full = prices_full.iloc[:, 1]
    spread_full = compute_spread(y_full, x_full, beta)
    z_full = compute_zscore(spread_full, w)
    signal_full = generate_signal(z_full)
    signal_test = signal_full.loc[test_s:test_e]

    # backtest on test period
    bt = run_backtest(prices_test, signal_test, beta,
                      capital=1.0, fee_rate=0.0005)

    # metrics on test period only
    market_test = market.loc[test_s:test_e]
    rf_test = rf.loc[test_s:test_e]
    m = summarize(bt['capital'], bt['returns'], rf_test, market_test)

    return {
        'train_start': train_s, 'train_end': train_e,
        'test_start': test_s, 'test_end': test_e,
        'status': 'traded',
        'beta': beta,
        'pvalue': coint['pvalue'],
        'half_life': hl,
        'window_w': w,
        'n_trades': bt['n_trades'],
        'returns': bt['returns'],
        'capital': bt['capital'],
        **m,
    }


def run_walk_forward(pair, start='2016-01-01', end='2026-01-01'):
    prices = load_pair(pair['ticker_y'], pair['ticker_x'], start, end)
    market = load_market('SPY', start, end)
    rf = load_rf(start, end)

    windows = get_windows()
    folds = [run_fold(prices, market, rf, *w) for w in windows]

    # aggregate only over traded folds
    traded = [f for f in folds if f['status'] == 'traded']
    n_traded = len(traded)

    if n_traded == 0:
        return {
            'folds': folds,
            'aggregate': None,
            'n_traded': 0,
            'n_total': len(folds),
        }

    # concatenate out-of-sample returns across folds
    all_returns = pd.concat([f['returns'] for f in traded]).sort_index()

    # reconstruct a single out-of-sample capital trajectory by compounding.
    # the initial capital is explicitly 1.0, declared through summarize's `base`
    # argument so the calculation does not rely on the invariant that
    # returns[0] == 0 holds for every fold's run_backtest.
    capital_oos = (1.0 + all_returns).cumprod()

    # aggregate metrics on the concatenated OOS sequence
    agg_m = summarize(capital_oos, all_returns, rf, market, base=1.0)

    aggregate = {
        'n_traded_folds': n_traded,
        'n_total_folds': len(folds),
        'total_return': float((1.0 + all_returns).prod()) - 1.0,
        'n_trades': int(sum(f['n_trades'] for f in traded)),
        **agg_m,
    }

    return {
        'folds': folds,
        'aggregate': aggregate,
        'n_traded': n_traded,
        'n_total': len(folds),
    }


def print_report(result, pair):
    # condensed per-fold and aggregate summary
    print(f'\n{"="*60}')
    print(f"Walk-Forward Report: {pair['label']}")
    print(f'{"="*60}')

    print(f"\n{'fold':<5}{'status':<12}{'pvalue':<10}{'beta':<10}"
          f"{'hl':<8}{'Sharpe':<9}{'trades':<8}")
    for i, f in enumerate(result['folds'], start=1):
        test_year = f['test_start'][:4]
        if f['status'] == 'skipped':
            pv = f.get('pvalue')
            pv_str = f'{pv:.3f}' if pv is not None else '-'
            reason = f['skip_reason']
            print(f"{test_year:<5}{'SKIP':<12}{pv_str:<10}"
                  f"{'-':<10}{'-':<8}{'-':<9}{'-':<8}  ({reason})")
        else:
            print(f"{test_year:<5}{'TRADED':<12}{f['pvalue']:<10.3f}"
                  f"{f['beta']:<10.3f}{f['half_life']:<8.0f}"
                  f"{f['sharpe']:<9.3f}{f['n_trades']:<8}")

    agg = result['aggregate']
    if agg is None:
        print('\nNo folds traded — no aggregate metrics.')
        return

    print(f"\n--- Aggregate (OOS concatenated over "
          f"{agg['n_traded_folds']}/{agg['n_total_folds']} folds) ---")
    print(f"total return  : {agg['total_return']:.4f}")
    print(f"ann return    : {agg['ann_return']:.4f}")
    print(f"ann vol       : {agg['ann_vol']:.4f}")
    print(f"sharpe        : {agg['sharpe']:.4f}")
    print(f"max drawdown  : {agg['max_drawdown']:.4f}")
    print(f"beta_m        : {agg['beta_m']:.4f}  (p={agg['beta_pvalue']:.3f})")
    print(f"alpha (daily) : {agg['alpha_daily']:.6f}  "
          f"(p={agg['alpha_pvalue']:.3f})")
    print(f"r_squared     : {agg['r_squared']:.4f}")
    print(f"total trades  : {agg['n_trades']}")


if __name__ == '__main__':
    # smoke test — change pair key here to test another
    from pairs import get_pair

    pair = get_pair('GSMS')
    result = run_walk_forward(pair, '2016-01-01', '2026-01-01')
    print_report(result, pair)

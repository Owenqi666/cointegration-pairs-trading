import numpy as np
import pandas as pd


def run_backtest(prices, signal, beta, capital=1.0, fee_rate=0.0005):
    # T+1 execution: today's signal is acted on next trading day
    # effective_signal[t] = signal[t-1] = position actually held on day t
    effective_signal = signal.shift(1).fillna(0)

    y = prices.iloc[:, 0].values.astype(float)
    x = prices.iloc[:, 1].values.astype(float)
    sig = effective_signal.values
    n = len(y)

    pnl = np.zeros(n)
    cost = np.zeros(n)

    # share counts, locked in at the moment of opening
    n_y = 0.0
    n_x = 0.0

    for i in range(1, n):
        prev_sig = sig[i-1]
        curr_sig = sig[i]

        # position change happens at the start of day i
        if curr_sig != prev_sig:
            # close old position first (if any)
            if prev_sig != 0:
                cost[i] += fee_rate * (n_y * y[i] + n_x * x[i])
                n_y = 0.0
                n_x = 0.0
            # open new position: size so that total notional = capital
            # and share ratio N_X / N_Y = beta (matches cointegrating vector)
            if curr_sig != 0:
                denom = y[i] + beta * x[i]
                n_y = capital / denom
                n_x = beta * capital / denom
                cost[i] += fee_rate * (n_y * y[i] + n_x * x[i])

        # PnL during day i: held position * close-to-close change in spread
        # spread move = ΔY - β * ΔX, signed by curr_sig (+1 long, -1 short)
        if curr_sig != 0:
            pnl[i] = curr_sig * (n_y * (y[i] - y[i-1]) - n_x * (x[i] - x[i-1]))

        pnl[i] -= cost[i]

    # cumulative capital trajectory
    capital_series = capital + np.cumsum(pnl)

    # daily returns relative to initial capital (no compounding)
    daily_returns = pnl / capital

    # number of position changes
    n_trades = int((effective_signal.diff().abs() > 0).sum())

    return {
        'capital': pd.Series(capital_series, index=prices.index),
        'returns': pd.Series(daily_returns, index=prices.index),
        'pnl': pd.Series(pnl, index=prices.index),
        'cost': pd.Series(cost, index=prices.index),
        'n_trades': n_trades,
    }


if __name__ == '__main__':
    # full pipeline smoke test — change pair key here to test another
    from pairs import load_smoke
    from cointegration import engle_granger, half_life
    from signal import compute_spread, compute_zscore, generate_signal

    prices, y, x, pair = load_smoke('GSMS')
    print(f"=== Smoke test: {pair['label']} ===")

    result = engle_granger(y, x)
    beta = result['beta']
    print('beta:', beta)
    print('cointegration p-value:', result['pvalue'])

    spread = compute_spread(y, x, beta)
    hl = half_life(spread)
    w = int(round(hl)) if not np.isnan(hl) else 30
    print('half-life:', hl)
    print('window w:', w)

    z = compute_zscore(spread, w)
    signal = generate_signal(z)
    print('n signal changes:', int((signal.diff().abs() > 0).sum()))

    bt = run_backtest(prices, signal, beta, capital=1.0, fee_rate=0.0005)

    print('\n=== Backtest results ===')
    print('final capital:', float(bt['capital'].iloc[-1]))
    print('total return:', float(bt['capital'].iloc[-1]) - 1.0)
    print('n_trades:', bt['n_trades'])
    print('total cost:', float(bt['cost'].sum()))
    print('max daily PnL:', float(bt['pnl'].max()))
    print('min daily PnL:', float(bt['pnl'].min()))
    print('PnL std:', float(bt['pnl'].std()))

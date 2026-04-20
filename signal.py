import numpy as np
import pandas as pd


def compute_spread(y, x, beta):
    # alpha is intentionally omitted: it shifts the mean but not the volatility,
    # which gets absorbed by the rolling mean in the z-score step
    return y - beta * x


def compute_zscore(spread, w):
    # rolling mean and std over window w
    mu = spread.rolling(window=w).mean()
    sigma = spread.rolling(window=w).std()
    z = (spread - mu) / sigma
    return z


def generate_signal(z, entry=2.0, exit_th=0.5, stop=3.0):
    # stateful loop: position depends on previous position and current z
    # state machine:
    #   flat   + z >  entry  -> short spread (-1)
    #   flat   + z < -entry  -> long spread  (+1)
    #   short  + |z| < exit  -> close (0)
    #   short  + |z| > stop  -> stop loss (0)
    #   long   + |z| < exit  -> close (0)
    #   long   + |z| > stop  -> stop loss (0)
    # no direct reversal: must close to flat before opening opposite side
    z_arr = z.values
    n = len(z_arr)
    signal = np.zeros(n)
    pos = 0

    for i in range(n):
        zi = z_arr[i]

        # nan during the warmup window, stay flat
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

    return pd.Series(signal, index=z.index)


if __name__ == '__main__':
    # smoke test — change pair key here to test another
    from pairs import load_smoke
    from cointegration import engle_granger, half_life

    prices, y, x, pair = load_smoke('GSMS')
    print(f"=== Smoke test: {pair['label']} ===")

    result = engle_granger(y, x)
    beta = result['beta']
    spread = compute_spread(y, x, beta)

    hl = half_life(spread)
    w = int(round(hl)) if not np.isnan(hl) else 30
    print('beta:', beta)
    print('half-life:', hl)
    print('window w:', w)

    z = compute_zscore(spread, w)
    print('\nz-score stats:')
    print('mean:', float(z.mean()))
    print('std :', float(z.std()))
    print('min :', float(z.min()))
    print('max :', float(z.max()))

    signal = generate_signal(z)
    print('\nsignal distribution:')
    print(signal.value_counts())

    # number of trades = number of position changes
    n_trades = int((signal.diff().abs() > 0).sum())
    print('n_trades:', n_trades)

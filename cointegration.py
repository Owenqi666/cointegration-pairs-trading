import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def adf_test(x):
    # wrapper around statsmodels adfuller, returns p-value only
    x = np.asarray(x).astype(float)
    x = x[~np.isnan(x)]
    result = adfuller(x, autolag='AIC')
    return result[1]


def hedge_ratio(y, x):
    # OLS: y = alpha + beta * x
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    alpha = float(model.params[0])
    beta = float(model.params[1])
    return alpha, beta


def engle_granger(y, x, threshold=0.05):
    # two-step cointegration test
    alpha, beta = hedge_ratio(y, x)

    # residuals from the cointegrating regression
    y_arr = np.asarray(y).astype(float)
    x_arr = np.asarray(x).astype(float)
    residuals = y_arr - alpha - beta * x_arr

    # ADF on residuals tests stationarity of the spread
    pvalue = adf_test(residuals)
    is_coint = pvalue < threshold

    return {
        'alpha': alpha,
        'beta': beta,
        'residuals': residuals,
        'pvalue': pvalue,
        'is_coint': is_coint,
    }


def half_life(spread, max_days=252):
    # OU discretization: Δz_t = λ * z_{t-1} + μ' + ε
    # mean reversion requires λ < 0
    z = np.asarray(spread).astype(float)
    z = z[~np.isnan(z)]

    z_lag = z[:-1]
    dz = np.diff(z)

    z_lag_const = sm.add_constant(z_lag)
    model = sm.OLS(dz, z_lag_const).fit()
    lam = float(model.params[1])

    # if lam >= 0 the spread does not mean-revert at all
    if lam >= 0:
        return np.nan

    hl = -np.log(2) / lam

    # reject half-lives longer than one trading year:
    # economically untradable even if statistically positive
    if hl > max_days:
        return np.nan

    return hl


if __name__ == '__main__':
    # smoke test — change pair key here to test another
    from pairs import load_smoke

    prices, y, x, pair = load_smoke('GSMS')
    print(f"=== Smoke test: {pair['label']} ===")

    print('\n=== ADF on raw prices (expect non-stationary) ===')
    print(f"{pair['ticker_y']} pvalue:", adf_test(y))
    print(f"{pair['ticker_x']} pvalue:", adf_test(x))

    print('\n=== Engle-Granger ===')
    result = engle_granger(y, x)
    print('alpha:', result['alpha'])
    print('beta :', result['beta'])
    print('pvalue:', result['pvalue'])
    print('is_coint:', result['is_coint'])

    print('\n=== Half-life ===')
    hl = half_life(result['residuals'])
    print('half-life (days):', hl)

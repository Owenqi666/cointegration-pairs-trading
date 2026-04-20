# Pairs Trading with Cointegration

A statistical arbitrage framework that identifies mean-reverting pairs of stocks, generates trading signals from standardized spread deviations, and evaluates performance through two complementary out-of-sample protocols. A **classical study** applies Walk-Forward validation to same-sector pairs where cointegration is expected to hold in equilibrium; an **extension study** applies rolling cointegration to an AI supply-chain pair where the relationship is theme-driven and classical methods are expected to fail.

## How to Run

```bash
git clone https://github.com/Owenqi666/pairs_trading
cd pairs_trading
pip install -r requirements.txt

python main.py       # Study 1: Walk-Forward on KO/PEP and GS/MS
python rolling.py    # Study 2: Rolling cointegration on NVDA/MSFT
```

Individual modules can also be run for their smoke tests (e.g. `python backtest.py`); the test pair is controlled by the `'GSMS'` key at the top of each smoke block and can be changed to `'KOPEP'` or `'NVDAMSFT'` without touching any other code, thanks to the central registry in `pairs.py`.

## Project Structure

```
pairs_trading/
├── data.py              # Price data loading (pair, market benchmark, ^IRX risk-free)
├── pairs.py             # Central registry of all pair metadata and smoke-test helper
├── cointegration.py     # ADF test, Engle-Granger two-step, OU half-life estimation
├── signal.py            # Spread construction, rolling z-score, state-machine signal
├── backtest.py          # Dollar-normalized position sizing, T+1 PnL, transaction costs
├── metrics.py           # Sharpe, max drawdown, CAPM-style market-neutrality regression
├── walk_forward.py      # 7-fold expanding-origin walk-forward orchestration
├── main.py              # Study 1 entry point (classical same-sector pairs)
└── rolling.py           # Study 2 entry point (rolling cointegration extension)
```

## Strategy Logic

Pairs trading exploits **mean reversion in the spread** between two cointegrated assets. Even when both prices are non-stationary random walks, a stable linear combination can remain stationary — the residual of this combination (the spread) oscillates around a long-run mean.

When the spread deviates significantly from its mean, the strategy opens a market-neutral position:

- Short the overvalued leg, long the undervalued leg
- Close when the spread reverts toward the mean
- Stop-loss if the spread continues to widen beyond a threshold

Because the two legs are hedged in dollar-normalized terms and weighted by the estimated cointegration coefficient, the portfolio's exposure to systematic market risk is theoretically eliminated. The strategy earns returns from the pair's relative mispricing, not from directional market movement.

## Mathematical Framework

### 1. Stationarity (Augmented Dickey-Fuller Test)

A time series is tested for a unit root through the regression

$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \epsilon_t$$

Null hypothesis

$$H_0: \gamma = 0 \quad \text{(unit root, non-stationary)}$$

Rejection at the chosen significance level indicates stationarity. Price series are typically $I(1)$ — non-stationary in levels but stationary in first differences.

### 2. Engle-Granger Cointegration

Two $I(1)$ series $Y_t$ and $X_t$ are cointegrated if there exists $\beta$ such that

$$Z_t = Y_t - \beta X_t \sim I(0)$$

**Step 1.** Estimate the hedge ratio by OLS

$$\hat{\beta} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$

**Step 2.** Apply ADF test to the residuals

$$\hat{\epsilon}_t = Y_t - \hat{\alpha} - \hat{\beta} X_t$$

Stationary residuals confirm cointegration and justify treating the spread as tradable.

### 3. Spread and Z-score

The spread is computed as

$$\text{Spread}_t = Y_t - \hat{\beta} X_t$$

Rolling mean and standard deviation over window $w$

$$\mu_t = \frac{1}{w} \sum_{i=t-w+1}^{t} \text{Spread}_i$$

$$\sigma_t = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (\text{Spread}_i - \mu_t)^2}$$

Z-score

$$z_t = \frac{\text{Spread}_t - \mu_t}{\sigma_t}$$

### 4. Signal Generation

Entry at 2 standard deviations, exit at 0.5, stop-loss at 3

$$\text{Signal}_t = \begin{cases} -1 & z_t > 2 \\ +1 & z_t < -2 \\ 0 & |z_t| < 0.5 \\ 0 & |z_t| > 3 \end{cases}$$

A signal of $-1$ shorts the spread (short $Y$, long $\hat{\beta} X$); $+1$ longs the spread. The state machine does not permit direct reversals — positions must close to flat before opening the opposite side.

### 5. Half-Life of Mean Reversion

Modeling the spread as an Ornstein-Uhlenbeck process

$$dZ_t = -\theta (Z_t - \mu) dt + \sigma dW_t$$

The discrete-time analog is estimated by regressing spread changes on lagged levels

$$\Delta Z_t = \lambda Z_{t-1} + \mu' + \epsilon_t$$

Half-life

$$\text{Half-Life} = -\frac{\ln 2}{\lambda}$$

Half-life informs the choice of rolling window: if the spread reverts in 10 days, a 60-day window dilutes the signal; if it reverts in 100 days, a 30-day window is too noisy. When the estimated $\lambda \geq 0$ (no reversion) or when the implied half-life exceeds 252 trading days, the pair is treated as non-tradable and the fold is skipped.

### 6. Position Sizing and PnL

Capital-normalized allocation: share counts are chosen so that total notional equals the capital $C$ and the share ratio $N_X / N_Y$ matches the cointegrating coefficient $\hat{\beta}$

$$N_Y = \frac{C}{Y_t + \hat{\beta} X_t}, \quad N_X = \frac{\hat{\beta} C}{Y_t + \hat{\beta} X_t}$$

This preserves the theoretical hedge ratio while preventing leverage from inflating when $\hat{\beta}$ deviates from $1$. Under the naive convention $N_Y = C/(2Y_t), N_X = \hat{\beta}C/(2X_t)$, a pair with $\hat{\beta} = 4$ would generate $2.5 \times$ gross notional and break dollar-neutrality.

Daily PnL with T+1 execution delay

$$\text{PnL}_t = s_{t-1} \cdot [N_Y (Y_t - Y_{t-1}) - N_X (X_t - X_{t-1})] - \text{Cost}_t$$

Transaction cost on position changes

$$\text{Cost}_t = f \cdot |s_t - s_{t-1}| \cdot (N_Y Y_t + N_X X_t)$$

with $f = 0.0005$.

### 7. Performance Metrics

Annualized return, geometric, used for cumulative performance reporting

$$R_{\text{ann}} = (1 + R_{\text{total}})^{252/T} - 1$$

Mean daily excess return, arithmetic, used for the Sharpe numerator

$$\bar{r} \cdot 252$$

Annualized volatility

$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \cdot \sqrt{252}$$

Sharpe ratio with live risk-free rate from `^IRX`

$$\text{Sharpe} = \frac{\bar{r} \cdot 252 - \bar{R_f}}{\sigma_{\text{ann}}}$$

The arithmetic annualization appears in the Sharpe numerator because Sharpe is defined in terms of expected excess return, which is arithmetic; the geometric annualization is used only for displaying total performance. Both quantities are reported in the per-pair tables and the difference between them is negligible at daily frequency for the return magnitudes observed in this study.

Maximum drawdown

$$\text{MDD} = \max_t \frac{\max_{s \le t} C_s - C_t}{\max_{s \le t} C_s}$$

### 8. Market-Neutrality Verification

CAPM-style regression on excess returns

$$r_t^{\text{strat}} - r_{f,t} = \alpha + \beta_m (r_t^{\text{mkt}} - r_{f,t}) + \epsilon_t$$

A market-neutral strategy should exhibit $\beta_m \approx 0$ both economically and statistically, and a low $R^2$ indicating that market movement explains little of the strategy's variance. This regression is the decisive test of whether the strategy earns pair-specific mispricing returns rather than residual directional exposure.

## Walk-Forward Validation

Pair selection and hedge ratio estimation must be done in-sample only to avoid lookahead bias. The 2016–2026 window is split into seven one-year out-of-sample test periods, each preceded by a three-year training window:

| Train Window | Test Window |
|--------------|-------------|
| 2016–2018    | 2019        |
| 2017–2019    | 2020        |
| 2018–2020    | 2021        |
| 2019–2021    | 2022        |
| 2020–2022    | 2023        |
| 2021–2023    | 2024        |
| 2022–2024    | 2025        |

For each fold:

1. Run Engle-Granger on the training window to estimate $\hat{\beta}$ and confirm cointegration. The threshold is relaxed to $p < 0.10$ in line with pairs-trading literature (Alexander 2008, Vidyamurthy 2004), reflecting the reduced statistical power of ADF on short three-year windows.
2. Compute half-life on the training window to set the rolling window $w$. Folds with $\lambda \geq 0$ or half-life $> 252$ days are skipped.
3. Apply the fixed $\hat{\beta}$ and $w$ to the test window, generating signals and PnL out-of-sample. Spread and z-score computation uses the combined train-test history for rolling warmup, but all signals and PnL are recorded only on test dates.
4. Aggregate across traded folds by concatenating out-of-sample returns into a single sequence and computing unified Sharpe, drawdown, and market-neutrality metrics.

## Rolling Cointegration

Study 2 replaces the static train-test split with a **daily rolling procedure** designed for pairs whose economic relationship is episodic rather than permanent. At each trading day $t$:

1. Fit OLS on the trailing 60-day window to obtain a time-varying hedge ratio $\hat{\beta}_t$.
2. Apply ADF to the window residuals, yielding a time-varying cointegration p-value $p_t$.
3. Standardize the current residual within the window to obtain $z_t$.

Trading is gated: when $p_t \geq 0.05$, the pair is treated as non-tradable and any open position is closed; when $p_t < 0.05$, the normal state machine applies. Position sizing locks $\hat{\beta}_t$ at the moment of opening, so each trade uses the hedge ratio estimated from the window ending at the entry day.

The stricter threshold of $0.05$ (compared with Study 1's $0.10$) partially compensates for the multiple-testing inflation inherent in running the ADF test on every single day.

## Study 1: Same-Sector Pairs (Classical Stat Arb)

Pairs are selected from same-industry stocks where common macro and industry factors dominate, leaving idiosyncratic spreads that revert to equilibrium.

| Sector | Pair | Role in Study |
|--------|------|---------------|
| Beverages | KO / PEP | Textbook baseline — the most-cited pair in stat arb literature, used to verify whether classical results still hold. |
| Investment Banking | GS / MS | Stress test — includes the 2020 COVID crash and the 2023 regional banking crisis, testing whether cointegration survives systemic shocks. |

## Study 2: AI Supply-Chain Pair (Extension)

The pair is selected along the AI value chain, where the economic link is **business dependence** rather than same-industry competition.

| Relationship | Pair | Role in Study |
|--------------|------|---------------|
| GPU supplier vs. AI hyperscaler | NVDA / MSFT | Boundary case — tests whether rolling cointegration can recover tradability when the underlying relationship is theme-driven (AI capex cycle) rather than equilibrium-driven. |

## Results

### Study 1: Walk-Forward Per-Fold Detail

**KO / PEP**

| Test Year | Status | p-value | $\hat{\beta}$ | Half-Life | Sharpe | Trades |
|-----------|--------|---------|---------------|-----------|--------|--------|
| 2019 | SKIP | 0.555 | — | — | — | — |
| 2020 | SKIP | 0.102 | — | — | — | — |
| 2021 | SKIP | 0.147 | — | — | — | — |
| 2022 | TRADED | 0.071 | 0.254 | 36 | −1.526 | 13 |
| 2023 | TRADED | 0.028 | 0.329 | 34 | −0.414 | 20 |
| 2024 | SKIP | 0.109 | — | — | — | — |
| 2025 | SKIP | 0.601 | — | — | — | — |

**GS / MS**

| Test Year | Status | p-value | $\hat{\beta}$ | Half-Life | Sharpe | Trades |
|-----------|--------|---------|---------------|-----------|--------|--------|
| 2019 | SKIP | 0.725 | — | — | — | — |
| 2020 | TRADED | 0.086 | 3.809 | 57 | +0.624 | 14 |
| 2021 | SKIP | 0.115 | — | — | — | — |
| 2022 | SKIP | 0.169 | — | — | — | — |
| 2023 | TRADED | 0.025 | 3.657 | 42 | −0.517 | 14 |
| 2024 | TRADED | 0.079 | 3.281 | 35 | −0.517 | 22 |
| 2025 | SKIP | 0.103 | — | — | — | — |

### Aggregate OOS Metrics (Both Studies)

| Metric | KO / PEP | GS / MS | NVDA / MSFT |
|--------|----------|---------|-------------|
| Tradable folds / days | 2 / 7 | 3 / 7 | 467 / 2455 (19.0%) |
| Total return | −1.92% | +9.11% | −13.09% |
| Ann. return | −0.97% | +2.95% | −1.40% |
| Ann. vol | 4.19% | 7.10% | 7.20% |
| Sharpe | −1.051 | −0.039 | −0.482 |
| Max drawdown | 7.34% | 6.42% | 20.67% |
| Total trades | 33 | 50 | 52 |

### Market-Neutrality Cross-Verification

Across three fundamentally different market regimes — consumer staples, investment banking, and AI infrastructure — the dollar-normalized hedge consistently eliminates market exposure both economically and statistically.

| Strategy | $\beta_m$ | $\beta$ p-value | $R^2$ |
|----------|-----------|-----------------|-------|
| KO / PEP (Walk-Forward) | −0.0054 | 0.577 | 0.0006 |
| GS / MS (Walk-Forward) | +0.0124 | 0.295 | 0.0015 |
| NVDA / MSFT (Rolling)   | −0.0066 | 0.411 | 0.0003 |

All three $\beta_m$ estimates are economically indistinguishable from zero; all three p-values fail to reject $H_0: \beta_m = 0$; all three $R^2$ values are below $0.002$, meaning market factor variation explains less than $0.2\%$ of strategy return variance. This is the strongest possible empirical evidence that the position-sizing and hedging implementation is correct — across pairs that otherwise differ dramatically in profitability, volatility, and cointegration behavior.

## Key Findings

**Finding 1 — Classical stat arb has weakened in the modern era.** Both KO/PEP ($p = 0.83$) and GS/MS ($p = 0.44$) fail the full-sample Engle-Granger test on 2016–2026 data, consistent with the hypothesis that textbook equilibrium relationships have eroded. KO/PEP's case is attributable to PepsiCo's strategic shift toward snacks, structurally decoupling its revenue drivers from Coca-Cola's pure beverage focus; GS/MS's case reflects increased idiosyncratic events (COVID, SVB) overwhelming the investment-banking common factor.

**Finding 2 — In-sample cointegration does not imply out-of-sample profitability.** The KO/PEP 2022 fold illustrates this most clearly: the training window (2019–2021) showed borderline cointegration ($p = 0.071$), yet the test year produced Sharpe $-1.53$ as the spread continued to diverge. This is a classic case of **spurious stationarity** — the training period's COVID-era consumer-behavior homogenization created transient statistical co-movement that did not persist. Walk-Forward validation is precisely the protocol that catches this failure mode, whereas full-sample or single-split testing would either obscure it or misattribute its cause.

**Finding 3 — Rolling cointegration quantifies "theme-driven" pair behavior but does not rescue it.** NVDA/MSFT passed the cointegration gate on only $19.0\%$ of trading days, providing a numerical measurement of how intermittent the mean-reverting relationship actually is. Yet even within these tradable windows, the strategy lost $13.1\%$ cumulatively and suffered a $20.7\%$ drawdown — trend resumption after brief cointegration episodes systematically destroyed positions opened at $|z| = 2$ before they could revert. This supports the thesis that **cointegration is a necessary but not sufficient condition for profitable pairs trading**, and rolling detection cannot compensate for an underlying economic relationship that is trend-based rather than equilibrium-based.

## Key Takeaways

- **Cointegration is an equilibrium statement, not a correlation statement.** Two highly correlated stocks can still have a non-stationary spread that diverges indefinitely. The Engle-Granger test distinguishes genuine equilibrium from spurious co-movement.
- **Same-sector pairs produce stable cointegration but thin margins.** When the test passes, the spread volatility is small, so realized Sharpe depends heavily on low transaction costs, disciplined execution, and a supportive risk-free rate environment.
- **Theme-driven pairs require dynamic frameworks, but dynamism alone is insufficient.** Rolling cointegration partially addresses the lack of permanent equilibrium, but cannot eliminate the boundary risk when a pair's true driver is a one-directional trend rather than mean reversion.
- **Market neutrality is not a byproduct of signal quality.** The position-sizing mechanics must be verified independently of profitability. A strategy can lose money while remaining rigorously hedged, and a profitable strategy can hide substantial directional exposure unless the $\beta_m$ regression is explicitly run.
- **Sharpe ratio is sensitive to the rate environment, not only to strategy alpha.** The GS/MS 2020 fold achieved Sharpe $+0.62$ in a zero-rate regime, while identical signal logic in 2023–2024 produced negative Sharpe due solely to the $\sim 5\%$ risk-free benchmark. Reporting only aggregate Sharpe across heterogeneous rate regimes obscures where the strategy genuinely generates alpha.

## Limitations

**Cointegration relationships can break permanently.** Corporate events such as mergers, spin-offs, and regulatory shocks can invalidate historical equilibrium. The Walk-Forward protocol detects this in retrospect through fold-level skipping, but does not predict it in advance.

**Pair universe is small and pre-selected.** Real stat arb desks scan hundreds of pairs; this project uses a curated list of economically motivated candidates, which biases results toward cases where cointegration is at least plausible.

**Daily-frequency data underestimates realistic Sharpe.** Professional pairs trading often operates at intraday frequencies where mean reversion is stronger and capital turnover is higher. With daily data and $f = 0.0005$, realized Sharpe is a conservative lower bound.

**No bid-ask spread or slippage modeling.** Transaction costs are modeled as a flat percentage; real execution would face wider spreads on the shorter leg and borrow costs on the short side, particularly for NVDA during periods of extreme momentum.

**Close-to-close PnL approximation.** Daily PnL is computed using close-to-close price changes attributed entirely to the current day's position. Under strict T+1 semantics, the overnight gap between close(t−1) and open(t) belongs to the previous day's position, not today's; the current formulation misattributes this component on trade days. For GS/MS with $89$ trades over 10 years and typical overnight gaps of $0.2\text{–}0.5\%$, cumulative error is estimated at $\pm 0.03$ to $\pm 0.10$ on $\$1$ initial capital, potentially $20\text{–}60\%$ of realized total return. This simplification follows Gatev, Goetzmann, and Rouwenhorst (2006) and is standard in academic pairs trading literature, but can meaningfully impact intraday or leverage-heavy variants.

**Multiple-testing bias in rolling cointegration.** Running the ADF test on every 60-day window for the full 10-year sample performs $\sim 2400$ hypothesis tests, inflating the false-positive rate relative to a single test. The stricter threshold of $0.05$ (versus Walk-Forward's $0.10$) partially compensates, but no formal Bonferroni or FDR adjustment is applied. A production implementation would correct for this explicitly.

**Threshold asymmetry between studies.** Study 1 uses $p < 0.10$ while Study 2 uses $p < 0.05$. This asymmetry reflects two different concerns: Study 1 compensates for the low statistical power of ADF on short three-year windows, while Study 2 compensates for the multiple-testing inflation of daily rolling tests. Both choices are grounded in method-specific reasoning rather than an attempt to maximize measured profitability, but the inconsistency does mean the two studies' tradable-fraction numbers are not directly comparable.

**Fixed strategy parameters.** Entry threshold (2.0), exit threshold (0.5), stop-loss threshold (3.0), and fee rate (0.0005) are held constant across all pairs and all folds. These parameters were not tuned to the data, which protects against overfitting but also means the reported results do not represent optimal parameter choices for any specific pair.

**Trade count semantics.** The `n_trades` metric counts **position changes** (flat → long, flat → short, or exit to flat), so a single round-trip (flat → ±1 → flat) contributes two to the count. This convention was chosen because it captures the number of transactions on which transaction costs are actually paid, but it should not be confused with "number of round-trip trades," which would be half.

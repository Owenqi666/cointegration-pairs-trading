# Pairs Trading with Cointegration

A statistical arbitrage framework that identifies mean-reverting pairs of stocks, generates trading signals from standardized spread deviations, and evaluates performance through Walk-Forward validation. Two parallel studies are conducted: a **classical study** on same-sector pairs where cointegration holds in equilibrium, and an **extension study** on AI supply-chain pairs where the relationship is theme-driven and rolling cointegration is required.

## Project Structure

```
pairs_trading/
├── data.py              # Price data loading and preprocessing
├── cointegration.py     # ADF and Engle-Granger tests
├── signal.py            # Spread, z-score, and signal generation
├── backtest.py          # Position sizing, PnL, transaction costs
├── metrics.py           # Sharpe, MDD, market-neutrality regression
├── walk_forward.py      # Rolling out-of-sample validation
├── main.py              # Classical study entry point
└── rolling.py           # Rolling cointegration extension
```

## Strategy Logic

Pairs trading exploits **mean reversion in the spread** between two cointegrated assets. Even when both prices are non-stationary random walks, a stable linear combination can remain stationary — the residual of this combination (the spread) oscillates around a long-run mean.

When the spread deviates significantly from its mean, the strategy opens a market-neutral position:

- Short the overvalued leg, long the undervalued leg
- Close when the spread reverts toward the mean

Because the two legs are hedged in dollar terms and weighted by the estimated cointegration coefficient, the portfolio's exposure to systematic market risk is theoretically eliminated. The strategy earns returns from the pair's relative mispricing, not from directional market movement.

## Mathematical Framework

### 1. Stationarity (Augmented Dickey-Fuller Test)

A time series is tested for a unit root through the regression

$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \epsilon_t$$

Null hypothesis

$$H_0: \gamma = 0 \quad \text{(unit root, non-stationary)}$$

Rejection at the 5% level indicates stationarity. Price series are typically $I(1)$ — non-stationary in levels but stationary in first differences.

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

A signal of $-1$ shorts the spread (short $Y$, long $\hat{\beta} X$); $+1$ longs the spread.

### 5. Half-Life of Mean Reversion

Modeling the spread as an Ornstein-Uhlenbeck process

$$dZ_t = -\theta (Z_t - \mu) dt + \sigma dW_t$$

The discrete-time analog is estimated by regressing spread changes on lagged levels

$$\Delta Z_t = \lambda Z_{t-1} + \mu' + \epsilon_t$$

Half-life

$$\text{Half-Life} = -\frac{\ln 2}{\lambda}$$

Half-life informs the choice of rolling window: if the spread reverts in 10 days, a 60-day window dilutes the signal; if it reverts in 100 days, a 30-day window is too noisy.

### 6. Position Sizing and PnL

Dollar-neutral allocation with total capital $C$

$$N_Y = \frac{C/2}{Y_t}, \quad N_X = \frac{(C/2) \cdot \hat{\beta}}{X_t}$$

Daily PnL with T+1 execution delay

$$\text{PnL}_t = s_{t-1} \cdot [N_Y (Y_t - Y_{t-1}) - N_X (X_t - X_{t-1})] - \text{Cost}_t$$

Transaction cost on position changes

$$\text{Cost}_t = f \cdot |s_t - s_{t-1}| \cdot (N_Y Y_t + N_X X_t)$$

with $f = 0.0005$.

### 7. Performance Metrics

Annualized return

$$R_{\text{ann}} = (1 + R_{\text{total}})^{252/T} - 1$$

Annualized volatility

$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \cdot \sqrt{252}$$

Sharpe ratio (risk-free rate from `^IRX`)

$$\text{Sharpe} = \frac{R_{\text{ann}} - R_f}{\sigma_{\text{ann}}}$$

Maximum drawdown

$$\text{MDD} = \max_t \frac{\max_{s \le t} C_s - C_t}{\max_{s \le t} C_s}$$

### 8. Market-Neutrality Verification

Regress strategy returns on market returns (SPY)

$$r_t^{\text{strategy}} = \alpha + \beta_m \cdot r_t^{\text{SPY}} + \epsilon_t$$

A market-neutral strategy should exhibit $\beta_m \approx 0$ and statistically significant $\alpha$. This regression directly tests whether the strategy earns from pair-specific mispricing rather than market direction.

## Walk-Forward Validation

Pair selection and hedge ratio estimation must be done **in-sample only** to avoid lookahead bias. The full 2016–2026 window is split into seven one-year out-of-sample periods, each preceded by a three-year training window:

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

1. Run Engle-Granger on the training window to estimate $\hat{\beta}$ and confirm cointegration
2. Compute half-life on the training window to set the rolling window $w$
3. Apply the fixed $\hat{\beta}$ and $w$ to the test window, generating signals and PnL out-of-sample

This protocol ensures that every trade is executed using information available strictly before the trade date.

## Study 1: Same-Sector Pairs (Classical Stat Arb)

Pairs are selected from same-industry stocks where common macro and industry factors dominate, leaving idiosyncratic spreads that revert to equilibrium.

| Sector | Pair |
|--------|------|
| Investment Banking | GS / MS |
| Beverages | KO / PEP |
| Payments | V / MA |
| Consumer Staples | PG / CL |
| Telecom | VZ / T |

The Engle-Granger framework is expected to hold well on these pairs because the underlying economic relationship is an equilibrium, not a trend.

## Study 2: AI Supply-Chain Pairs (Extension)

Pairs are selected along the AI value chain, where the economic link is **business dependence** rather than same-industry competition:

| Relationship | Pair |
|--------------|------|
| GPU supplier vs. AI hyperscaler | NVDA / MSFT |
| GPU supplier vs. AI hyperscaler | NVDA / META |
| Semiconductor equipment vs. foundry | ASML / TSM |

These pairs are driven by a common theme (AI capex cycle) rather than an equilibrium relationship. Static Engle-Granger cointegration is expected to fail on most of the 2016–2026 window because one leg typically outruns the other in trend phases (NVDA's 2023–2024 rally being the clearest example).

**Rolling cointegration** is applied instead: the Engle-Granger test is re-run on a moving 60-day window, and trading is enabled only when the test passes at the 5% level within the current window. This allows the strategy to opportunistically capture mean-reverting episodes without assuming a permanent equilibrium.

## Key Takeaways

- **Cointegration is an equilibrium statement, not a correlation statement.** Two highly correlated stocks can still have a non-stationary spread that diverges indefinitely. The Engle-Granger test distinguishes genuine equilibrium relationships from spurious co-movement.
- **Same-sector pairs produce stable cointegration but thin margins.** The spread volatility is small, so realized Sharpe depends heavily on low transaction costs and disciplined execution.
- **Theme-driven pairs require dynamic frameworks.** Static cointegration fails on NVDA / MSFT because the AI capex cycle created a multi-year trend divergence. Rolling cointegration recovers some tradability by restricting activity to windows where stationarity empirically holds.
- **Market-neutrality is not automatic.** The $\beta_m$ regression against SPY reveals whether the hedge ratio actually neutralizes market exposure or whether residual directional risk remains.

## Limitations

- **Cointegration relationships can break permanently.** Corporate events (mergers, spin-offs, regulatory shocks) can invalidate historical equilibrium. The Walk-Forward protocol detects this in retrospect but does not predict it.
- **Pair universe is small and pre-selected.** Real stat arb desks scan hundreds of pairs; this project uses a curated list of economically motivated candidates, which biases results toward cases where cointegration is plausible.
- **Daily data underestimates realistic Sharpe.** Professional pairs trading often operates at intraday frequencies where mean reversion is stronger and capital turnover is higher. With daily data and $f = 0.0005$, realized Sharpe is a conservative lower bound.
- **No bid-ask spread or slippage modeling.** Transaction costs are modeled as a flat percentage; real execution would face wider spreads on the shorter leg and borrow costs on the short side.
- **Rolling cointegration introduces multiple testing bias.** Running the ADF test on every 60-day window inflates false positive rates. In a production setting, this would be corrected with Bonferroni or FDR adjustment.

# quant_statarb

A statistical arbitrage pairs trading project. The pipeline goes data, then cointegration, then signals, then backtester, then analytics, built up across four notebooks and the `src/` modules they call into.

The core idea: find two assets whose prices move together over time (cointegrated), trade the spread between them when it stretches too far from its mean, and exit when it reverts. Market-neutral, so the return should be uncorrelated with the overall market.

## src/

**`data_loader.py`** downloads adjusted close prices via yfinance, caches each ticker as parquet under `data/cache/`, forward-fills gaps of 1 day, and drops any ticker with more than 5% missing values. Also has `compute_returns` (log or simple) and `get_universe`, which returns predefined ticker lists for `banks`, `energy`, `etf_pairs`, or `all`.

**`cointegration.py`** has the statistical core: an ADF test wrapper, an OLS hedge ratio estimator, the two-step Engle-Granger cointegration test, half-life of mean reversion, the Hurst exponent, `screen_pairs` to test every pair in a universe at once (with a Bonferroni correction option for multiple testing), and `rolling_hedge_ratio` for a time-varying beta refit on a trailing window.

**`strategy.py`** turns a spread into a position. `rolling_zscore` computes the z-score of the spread. `generate_signals_stateless` maps z directly to a position (simple, but can't hold between entry and exit). `generate_signals_stateful` walks through the z-score maintaining state, so it correctly holds a position between entry and exit thresholds and adds a stop loss.

**`backtester.py`** builds the spread from a frozen or walk-forward beta, runs the stateful signals, computes gross P&L (position shifted one bar to avoid look-ahead), transaction costs, and the net equity curve. Supports three sizing modes: `unit` (just ±1), `dollar` (dollar-neutral notional), and `vol` (volatility targeted). Also extracts a trade log from the position series.

**`analytics.py`** computes the usual performance metrics (Sharpe, Sortino, max drawdown, Calmar), rolling Sharpe, trade statistics (hit rate, profit factor, expectancy), and a `sweep_sharpe` helper for parameter grids.

## The notebooks

**`01_data.ipynb`**: pulls prices for banks and the SPY/IVV pair, and immediately shows why cointegration testing matters. Even though SPY and IVV track the same index, the spread has a clear negative trend post-2016, mostly down to the expense ratio gap (IVV at 0.03% vs SPY at 0.0945%). Two ETFs tracking the same thing does not mean they're cointegrated.

**`02_stationarity.ipynb`**: sanity-checks the ADF test, half-life, and Hurst exponent against synthetic series with known properties. White noise and a mean-reverting AR(1) both come back stationary (p=0.00), the random walk doesn't (p=0.26). For phi=0.9, the theoretical half-life is 6.58 periods against an empirical 6.91, close enough to trust the estimator.

**`03_cointegration.ipynb`**: runs the real tests. MS and GS, despite being highly price-correlated, are not cointegrated (p=0.99 on the naive ADF, 0.34 on the custom Engle-Granger implementation). EWA/EWC comes out as the standout candidate: Engle-Granger p=0.064, custom p=0.018, half-life around 30 days, a plausible economic story (both commodity-driven economies). Screening a broad sector-by-sector universe with Bonferroni correction turns up three in-sample survivors (EWC/EWA, CRM/ADBE, UNP/NSC), but none of them hold up out-of-sample, a reminder that cointegration found in-sample is not a guarantee, and that testing many pairs raises the bar you need to clear.

**`04_test_strategy.ipynb`**: builds and stress-tests the full strategy on EWA/EWC.
- In-sample (2015-2020) equity curve: gross 7.98, net 7.17 (starting from 1.0). Out-of-sample (2021-2024) with the frozen beta: gross 4.09, net 3.25, weaker but still profitable, even though the pair's out-of-sample p-value (0.20) no longer clears the cointegration bar. Half-life drifted from about 14 to 44 days out-of-sample, evidence of a slowing but not dead relationship.
- Breakeven cost is 50 bps: past that, the in-sample strategy's net P&L goes negative.
- Normalized to capital deployed, the in-sample strategy returns 3.3% annually on a ~$31 capital base. Modest, but this is a dollar-neutral, market-neutral strategy, which is the point.
- Sizing modes compared: unit sizing gets 21.2% total return (Sharpe 0.83), dollar-neutral 23.1% (Sharpe 0.85), vol targeting 16.5% (Sharpe 0.72). Vol targeting rebalances daily as volatility estimates shift, which racks up transaction costs and drags the Sharpe down.
- Walk-forward beta (252-bar lookback, refit every 21 bars) drifted between 0.115 and 0.796 out-of-sample, versus a frozen beta of 0.729. Walk-forward actually underperforms the frozen beta out-of-sample (3.6% vs 4.6%), most likely because it's re-estimating on a period where the relationship had already weakened.
- A parameter sweep over entry z-score and rolling window gives a Sharpe range of 0.36 to 1.05 across the grid, median 0.79. The chosen defaults (window=60, entry_z=2.0) land at 0.83, right around the median, which is a decent sign the defaults weren't cherry-picked to flatter the backtest.

## Setup

```
pip install -r requirements.txt
```

Run the notebooks in order (01 through 04). Each one caches the prices it downloads under `data/cache/` so re-runs are fast.

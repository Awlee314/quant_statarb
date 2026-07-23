import pandas as pd
import numpy as np
from backtester import run_backtest


def performance_summary(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute standard performance metrics from a daily return series.

    Returns a dict with:
      - total_return, annualized_return, annualized_vol
      - sharpe, sortino, max_drawdown, calmar
      - n_periods, n_years

    Assumes returns are simple (not log) and arithmetic.
    Risk-free rate is annual; convert before subtracting from daily returns.
    """
    r = daily_returns.dropna()
    n = len(r)
    n_years = len(r) / periods_per_year
    cumulative_return = (1+r).cumprod()
    total_return = cumulative_return.iloc[-1] - 1
    annualized_return = (1+total_return)**(1/n_years) - 1
    annualized_vol = r.std() * np.sqrt(periods_per_year)
    rf_daily = risk_free_rate / periods_per_year
    sharpe = (r - rf_daily).mean() / r.std() * np.sqrt(periods_per_year)
    sd_down = np.sqrt((r[r < 0]**2).sum() / n)
    sortino = (r - rf_daily).mean() / (sd_down) * np.sqrt(periods_per_year) if sd_down != 0 else np.nan
    running_max = cumulative_return.cummax()
    dd = (cumulative_return - running_max) / running_max
    max_drawdown = dd.min()
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'n_periods': n,
        'n_years': n_years
    }

def drawdown_series(cum_returns: pd.Series) -> pd.Series:
    """
    Percentage drawdown from running peak, at each point in time.
    Returns values <= 0, where 0 means at a new high.
    """
    running_peak = cum_returns.cummax()
    drawdown = (cum_returns - running_peak) / running_peak
    return drawdown

def rolling_sharpe(
    daily_returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """Rolling annualized Sharpe over a trailing window."""
    rf_daily = risk_free_rate / periods_per_year
    diff = daily_returns - rf_daily
    roll_mean = diff.rolling(window).mean()
    roll_std = diff.rolling(window).std()

    return (roll_mean / roll_std) * np.sqrt(periods_per_year)

def trade_statistics(trades: pd.DataFrame, include_open: bool = False) -> dict:
    """
    Trade-level statistics from a trade log with a `trade_pnl` column.

    hit_rate       fraction of trades with positive P&L
    avg_win        mean P&L of winning trades
    avg_loss       mean P&L of losing trades (negative)
    win_loss_ratio |avg_win / avg_loss|
    profit_factor  gross profits / gross losses (>1 is profitable)
    expectancy     mean P&L per trade
    """

    if 'trade_pnl' not in trades.columns:
        raise ValueError("trades has no 'trade_pnl' column, pass net_pnl to extract trades.")
    
    # If we include open then all trades
    # If not then we exclude all trades that are still open
    t = trades if include_open else trades[~trades['still_open']]

    pnl = t['trade_pnl']

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = -losses.sum() # make positive

    avg_win = wins.mean() if len(wins) != 0 else np.nan
    avg_loss = losses.mean() if len(losses) != 0 else np.nan

    return {
        'n_trades': len(t),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'hit_rate':  len(wins)/ len(t) if len(t) != 0 else np.nan,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': abs(avg_win / avg_loss) if len(losses) != 0 and avg_loss != 0 else np.nan,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else np.inf,
        'expectancy': pnl.mean(),
        'avg_holding_period': t['holding_period'].mean()
    }

def sweep_sharpe(prices, y, x, beta, **kwargs) -> float:
    """Sharpe for one parameter configuration. No capital normalization
    needed, Sharpe is invariant to the scale of the P&L series."""
    r = run_backtest(prices, y, x, beta, **kwargs)
    pnl = r['equity_curve']['net_pnl']
    return (pnl.mean() / pnl.std()) * np.sqrt(252) if pnl.std() > 0 else np.nan
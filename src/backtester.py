from strategy import rolling_zscore, generate_signals_stateful, rolling_zscore_walkforward
from cointegration import rolling_hedge_ratio
import pandas as pd
import numpy as np


def extract_trades(positions: pd.Series,
                    net_pnl: pd.Series | None = None,
                    still_open: bool = True) -> pd.DataFrame:
    """
    Walk a position series and extract round-trip trades.

    A trade opens when position goes from 0 to ±1, and closes when it
    returns to 0. Returns a DataFrame with one row per completed trade:
      - entry_date, exit_date
      - direction (+1 long spread, -1 short spread)
      - holding_period (in bars)

    Open positions at the end of the series (never closed) are flagged
    If net_pnl is provided, adds a `trade_pnl` column: the sum of net P&L
    over each trade's holding window.
    """

    trade_info = []
    prev_val = 0
    entry_date = None
    for date, val in positions.items():
        if prev_val == 0 and val != 0:
            # If not in position
            entry_date = date
        if prev_val != 0 and val == 0:
            # Close the position
            holding = len(positions.loc[entry_date:date])-1
            # Gives us the length of trading days between entry and exit
            row = {'entry_date': entry_date, 
                               'exit_date': date,
                               'direction': int(np.sign(prev_val)),
                                'holding_period': holding,
                                'still_open': False}
            if net_pnl is not None:    
                index = net_pnl.index
                index_entry = index.get_loc(entry_date)
                index_exit = index.get_loc(date)
                trade_pnl = net_pnl.iloc[index_entry +1 : index_exit + 2].sum()
                row['trade_pnl'] = trade_pnl
            trade_info.append(row)
        prev_val = val
    
    # Note that we only close when going to zero.
    # This could have consequences if we ever go from say 1 to -1 instantly
    # as the position would not close. However, this is an almost impossible instance as stock prices
    # do not shift two standard deviations instantly.
    if prev_val != 0 and still_open:
        # Wish to count open positions as closed at the end
        last_date = positions.index[-1]
        holding = len(positions.loc[entry_date:last_date])-1

        row = {'entry_date': entry_date, 
                               'exit_date': last_date,
                               'direction': int(np.sign(prev_val)),
                                'holding_period': holding,
                                'still_open': True}
        if net_pnl is not None:    
            index = net_pnl.index
            index_entry = index.get_loc(entry_date)
            index_exit = index.get_loc(last_date)
            trade_pnl = net_pnl.iloc[index_entry +1 : index_exit + 2].sum()
            row['trade_pnl'] = trade_pnl
        trade_info.append(row)
    df = pd.DataFrame(trade_info)
    return df

def build_spread(prices: pd.DataFrame, y: str, x: str, beta: float) -> pd.Series:
    """
    Construct the spread s = y - beta * x using a FIXED hedge ratio.
    The beta is estimated in-sample and frozen; this function just applies it.
    Returns the spread series.
    """
    # y,x are the ticker symbols
    spread = prices[y] - beta * prices[x]
    return spread


def compute_pnl(
    positions: pd.Series,
    spread: pd.Series,
) -> pd.Series:
    """
    Compute daily gross P&L (before costs) from positions and a spread.

        pnl_t = position_{t-1} * (spread_t - spread_{t-1})

    The position is shifted forward one bar so that the position decided
    at the close of t-1 earns the spread change over t-1 -> t. This is
    what prevents look-ahead bias.

    Returns a daily P&L series (in spread units / dollars per 1 unit of y).
    """
    
    # spread_t - spread_{t-1}
    spread_diff = spread.diff()

    # Shift positions to use yesterdays position to decided PnL today.
    # Prevents look-ahead bias
    cur_positions = positions.shift(1)

    # Compute daily PnL
    PnL = cur_positions * spread_diff
    
    return PnL


def compute_costs(
    positions: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    beta: float | pd.Series,
    cost_bps: float = 5.0,
) -> pd.Series:
    """
    Per-bar transaction costs, charged on the two-leg dollar notional.

    When the position changes by Δ units, you trade Δ units of the spread,
    which means trading Δ * price_y dollars of y and Δ * beta * price_x
    dollars of x. Total notional per unit = price_y + beta * price_x.

    cost_t = cost_bps/10000 * |Δposition_t| * (price_y_t + beta * price_x_t)

    Returns a daily cost series (>= 0).
    """
    # Shift by one to have costs incur when decision executes
    traded_units = positions.shift(1).diff().abs()
    notional_per_unit = price_y + beta * price_x        # dollar value of one spread unit
    cost = (cost_bps / 10000) * traded_units * notional_per_unit
    return cost

def build_equity_curve(
    gross_pnl: pd.Series,
    costs: pd.Series,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Combine gross P&L and costs into a net equity curve.

    net_pnl_t = gross_pnl_t - costs_t
    equity_t  = initial_capital + cumulative sum of net_pnl

    Returns a DataFrame with columns: gross_pnl, costs, net_pnl,
    cum_gross, cum_net (the gross and net equity curves).
    """
    net_pnl = (gross_pnl - costs).fillna(0)
    cum_net = initial_capital + net_pnl.cumsum()
    cum_gross = initial_capital + gross_pnl.cumsum()

    df = pd.DataFrame({'gross_pnl': gross_pnl, 
                       'costs': costs,
                       'net_pnl': net_pnl,
                       'cum_gross': cum_gross,
                       'cum_net': cum_net,
                       })
    return df


def run_backtest(
    prices: pd.DataFrame,
    y: str,
    x: str,
    beta: float,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    cost_bps: float = 5.0,
    initial_capital: float = 1.0,
    sizing: str = 'unit',
    target_notional: float = 10000.0,
    target_daily_vol: float = 100.0
) -> dict:
    """
    Full single-pair backtest pipeline:
      1. build spread from frozen beta
      2. rolling z-score
      3. stateful signals
      4. gross P&L (position shifted one bar)
      5. costs
      6. equity curve
      7. trade log

    Returns a dict with: spread, zscore, positions, equity_curve (DataFrame),
    trades (DataFrame).
    """

    spread = build_spread(prices,y,x,beta)
    zscore = rolling_zscore(spread, window=window)
    signals = generate_signals_stateful(zscore, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)

    if sizing == 'unit':
        # Unit sizing
        positions = signals
    elif sizing == 'dollar':
        # Use dollar neutral sizing
        positions = size_dollar_neutral(signals, prices[y], prices[x], beta, target_notional)
    elif sizing == 'vol':
        # Volatility sizing
        positions = size_vol_target(signals, prices[y], prices[x], beta,
                                target_daily_vol, vol_window=window)
    else:
        raise ValueError("The provided sizing option is not accepted. Sizing options are" \
        ": unit, dollar, vol")

    PnL = compute_pnl(positions, spread)
    costs = compute_costs(positions, prices[y], prices[x], beta, cost_bps=cost_bps).fillna(0)
    equity = build_equity_curve(PnL, costs, initial_capital=initial_capital)

    dictionary = {
        'spread': spread,
        'zscore': zscore,
        'positions': positions,
        'equity_curve': equity,
        'trades': extract_trades(positions, net_pnl=equity['net_pnl'])

    }

    return dictionary


def compute_returns_series(
    net_pnl: pd.Series,
    capital: float,
) -> dict:
    """
    Normalize a net dollar P&L series into returns.
    Caller supplies the P&L (from compute_pnl or compute_pnl_walkforward)
    and the capital base, so this works for both fixed and walk-forward betas.
    """
    daily_return = net_pnl / capital
    cumulative = (1 + daily_return.fillna(0)).cumprod()
    return {
        'daily_return': daily_return,
        'capital_used': capital,
        'cum_return': cumulative,
        'total_return': cumulative.iloc[-1] - 1,
    }
    

def size_dollar_neutral(
    signals: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    beta: float,
    target_notional: float = 10000.0,
) -> pd.Series:
    """
    Convert ±1 signals into dollar-neutral sized positions.

    Each active position targets `target_notional` dollars of gross
    two-leg exposure. The number of spread units held is:

        units_t = target_notional / (price_y_t + beta * price_x_t)

    The sized position is sign_t * units_t, where sign comes from the
    signal. Flat signals (0) produce 0 units.

    Returns a Series of sized positions (continuous, in spread units).
    """

    units = target_notional / (price_y + beta * price_x)

    positions_sized = (units * signals).fillna(0)

    return positions_sized 

def size_vol_target(
    signals: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    betas: float | pd.Series,
    target_daily_vol: float = 100.0,
    vol_window: int = 60,

) -> pd.Series:
    """
    Convert ±1 signals into volatility-targeted sized positions.

    Sizes each position inversely to the volatility of the spread's true
    daily P&L per unit:

        true_change_t = Δy_t - beta_{t-1} * Δx_t
        units_t       = target_daily_vol / rolling_std(true_change)_{t-1}

    Using leg changes with the LAGGED beta (rather than spread.diff())
    means the volatility estimate is not contaminated by discontinuities
    in the spread when the hedge ratio is re-estimated. Works for both a
    constant beta (float) and a time-varying beta (Series).

    The rolling vol is lagged one bar to avoid look-ahead.
    """
    beta_lag = betas.shift(1) if isinstance(betas, pd.Series) else betas
    
    true_change = price_y.diff() - beta_lag * price_x.diff()
    # Compute rolling std over vol_window and shift
    # Note the volatility is the standard deviation of the spread CHANGES
    rolling_vol_lag = true_change.rolling(vol_window).std().shift(1)
    # Add a floor so small volatility does not explode units
    rolling_vol_lag = rolling_vol_lag.clip(lower=rolling_vol_lag.median()*0.1)
    # Scale units according to how much volatility
    # If large volatility we have less units, vice versa
    units = target_daily_vol / rolling_vol_lag

    positions_sized = (signals * units).fillna(0)
    # Fill NaN spots with 0

    return positions_sized



def compute_pnl_walkforward(
    positions: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    betas: pd.Series,
) -> pd.Series:
    """
    Daily gross P&L with a time-varying hedge ratio.

        pnl_t = position_{t-1} * [Δy_t - beta_{t-1} * Δx_t]

    Uses the beta that was IN EFFECT while the position was held
    (beta_{t-1}), not the newly-estimated beta, so that changes in the
    hedge ratio don't create phantom P&L. 
    """
    dy = price_y.diff()
    dx = price_x.diff()

    # Need the beta that was in effect in the current interval
    # beta_{t-1} at time t
    beta_lag = betas.shift(1)

    pnl = positions.shift(1) * (dy - beta_lag * dx)

    return pnl

def run_backtest_walkforward(
    prices: pd.DataFrame,
    y: str,
    x: str,
    lookback: int = 252,
    refit_every: int = 21,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    cost_bps: float = 5.0,
    initial_capital: float = 1.0,
    sizing: str = 'unit',
    target_notional: float = 10000.0,
    target_daily_vol: float = 100.0
) -> dict:
    """
    Walk-forward backtest: beta re-estimated on a trailing window.
    Returns dict with betas, spread, zscore, positions, equity_curve, trades.
    """
    betas = rolling_hedge_ratio(prices[y], prices[x], lookback, refit_every)
    spread = prices[y] - betas * prices[x]

    zscore = rolling_zscore_walkforward(prices[y], prices[x], betas, window=window)

    signals = generate_signals_stateful(zscore, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)

    if sizing == 'unit':
        # Unit sizing
        positions = signals
    elif sizing == 'dollar':
        # Use dollar neutral sizing
        positions = size_dollar_neutral(signals, prices[y], prices[x], betas, target_notional)
    elif sizing == 'vol':
        # Use volatility sizing
        positions = size_vol_target(signals, prices[y], prices[x], betas,
                                target_daily_vol, vol_window=window)
    else:
        raise ValueError("The provided sizing option is not accepted. Sizing options are" \
        ": unit, dollar, vol")

    PnL = compute_pnl_walkforward(positions, prices[y], prices[x], betas)
    # Note we have not included the
    # rebalancing costs from beta changes. This is a simplification of the process.
    costs = compute_costs(positions, prices[y], prices[x], betas, cost_bps=cost_bps)
    equity = build_equity_curve(PnL, costs, initial_capital=initial_capital)


    dictionary = {
        'betas' : betas,
        'spread': spread,
        'zscore': zscore,
        'positions': positions,
        'equity_curve': equity,
        'trades': extract_trades(positions, net_pnl=equity['net_pnl'])

    }

    return dictionary
from strategy import rolling_zscore, generate_signals_stateful
import pandas as pd


def extract_trades(positions: pd.Series, still_open: bool = True) -> pd.DataFrame:
    """
    Walk a position series and extract round-trip trades.

    A trade opens when position goes from 0 to ±1, and closes when it
    returns to 0. Returns a DataFrame with one row per completed trade:
      - entry_date, exit_date
      - direction (+1 long spread, -1 short spread)
      - holding_period (in bars)

    Open positions at the end of the series (never closed) are flagged
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
            trade_info.append({'entry_date': entry_date, 
                               'exit_date': date,
                               'direction': prev_val,
                                'holding_period': holding,
                                'still_open': False})
        prev_val = val
    
    # Note that we only close when going to zero.
    # This could have consequences if we ever go from say 1 to -1 instantly
    # as the position would not close. However, this is an almost impossible instance as stock prices
    # do not shift two standard deviations instantly.
    if prev_val != 0 and still_open:
        # Wish to count open positions as closed at the end
        last_date = positions.index[-1]
        holding = len(positions.loc[entry_date:last_date])-1
        trade_info.append({'entry_date': entry_date, 
                               'exit_date': last_date,
                               'direction': prev_val,
                                'holding_period': holding,
                                'still_open': True})
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
    beta: float,
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
    positions = generate_signals_stateful(zscore, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)
    PnL = compute_pnl(positions, spread)
    costs = compute_costs(positions, prices[y], prices[x], beta, cost_bps=cost_bps)

    dictionary = {
        'spread': spread,
        'zscore': zscore,
        'positions': positions,
        'equity_curve': build_equity_curve(PnL, costs, initial_capital=initial_capital),
        'trades': extract_trades(positions)

    }

    return dictionary
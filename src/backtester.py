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
        # Use volatility sizing
        positions = size_vol_target(signals, spread, target_daily_vol, vol_window=window)
    else:
        raise ValueError("The provided sizing option is not accepted. Sizing options are" \
        ": unit, dollar, vol")

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


def compute_returns_series(
    positions: pd.Series,
    spread: pd.Series,
    price_y: pd.Series,
    price_x: pd.Series,
    beta: float,
    costs: pd.Series,
    capital: float | None = None,
) -> dict:
    """
    Convert dollar P&L into a return series normalized by capital.

    Daily dollar P&L (net) = position_{t-1} * Δspread_t - cost_t
    Daily return          = net_dollar_pnl_t / capital

    If `capital` is None, default to the peak two-leg notional over the
    window: max(price_y + beta * price_x). This represents a fixed capital
    allocation to the strategy (Option 2 — return on allocated capital,
    including idle-cash drag on flat days).

    Returns a dict with:
      - daily_return: pd.Series
      - capital_used: float (the C that was divided by)
      - cum_return: pd.Series, the compounded equity curve (1+r).cumprod()
      - total_return: float, final cum_return - 1
    """
    if capital is None:
        # Set capital to max two-leg notional (fixed capital)
        capital = (price_y + beta * price_x).max()
    
    Net_PnL = compute_pnl(positions, spread) - costs

    daily_return = Net_PnL / capital

    cmpd = (1+daily_return.fillna(0)).cumprod()

    # store the total returns
    total_returns = cmpd.iloc[-1] - 1

    dictionary = {
        'daily_return': daily_return,
        'capital_used': capital,
        'cum_return': cmpd,
        'total_return': total_returns
    }
    return dictionary
    

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

    positions_sized = units * signals

    return positions_sized 

def size_vol_target(
    signals: pd.Series,
    spread: pd.Series,
    target_daily_vol: float = 100.0,
    vol_window: int = 60,
) -> pd.Series:
    """
    Convert ±1 signals into volatility-targeted sized positions.

    Sizes each position inversely to the spread's rolling volatility so
    that each trade targets `target_daily_vol` dollars of daily P&L
    volatility:

        units_t = target_daily_vol / rolling_std(Δspread)_{t-1}

    The rolling vol is LAGGED one bar (shift(1)) to avoid look-ahead:
    the size at t uses volatility known through t-1.

    Returns a Series of sized positions (continuous, in spread units).
    """

    # Compute rolling std over vol_window and shift
    # Note the volatility is the standard deviation of the spread CHANGES
    rolling_vol_lag = spread.diff().rolling(vol_window).std().shift(1)
    # Add a floor so small volatility does not explode units
    rolling_vol_lag = rolling_vol_lag.clip(lower=rolling_vol_lag.median()*0.1)
    # Scale units according to how much volatility
    # If large volatility we have less units, vice versa
    units = target_daily_vol / rolling_vol_lag

    positions_sized = (signals * units).fillna(0)
    # Fill NaN spots with 0

    return positions_sized
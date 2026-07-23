import pandas as pd
import numpy as np


def rolling_zscore(
        spread: pd.Series,
        window: int = 60,
) -> pd.Series:
    """
    Compute the rolling z-score of a spread.

        z_t = (spread_t - rolling_mean_t) / rolling_std_t

    The rolling mean and std use a trailing window. To avoid look-ahead
    bias, the statistics at time t use data up to and including t, but the
    SIGNAL derived from z will be shifted forward one bar before being
    applied to returns (handled later in the backtester).

    Returns a Series of z-scores, same index as spread, with NaN for the
    first `window` periods (insufficient data).
    """
    rolling_mean_t = spread.rolling(window).mean()
    rolling_std_t = spread.rolling(window).std()
    # By default pandas includes current bar as part of of window size

    z_t = (spread - rolling_mean_t) / rolling_std_t

    return z_t

def generate_signals_stateless(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """
    First-pass STATELESS signal: position is a direct function of current z.

      z > +entry_z  -> -1 (short spread)
      z < -entry_z  -> +1 (long spread)
      |z| < exit_z  ->  0 (flat)
      otherwise     ->  hold previous? (stateless can't do this cleanly)

    Returns a Series of target positions in {-1, 0, +1}.

    NOTE: This is a simplified first pass. It does NOT correctly handle the
    "hold position between entry and exit thresholds" behavior, that
    requires state, which we add in Phase 3B. This version is for
    understanding the z-to-position mapping only.
    """
    # Series of all zeroes initially
    signal = pd.Series(0, index=z.index)
    # Short the spread
    signal[z > entry_z] = -1
    # Long the spread
    signal[z < -entry_z] = 1
    # Note that this marks the holding area between entry and exit as 0
    # This is the issue of making a stateless signal generator which will
    # be addressed later.
    return signal


def generate_signals_stateful(
    z: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> pd.Series:
    """
    Stateful signal: walk through z-scores maintaining current position,
    applying entry/exit/stop transitions. Correctly holds a position
    between entry and exit thresholds.

    Returns a Series of positions in {-1, 0, +1}, same index as z.
    """
    position = 0
    positions_list = []
    for val in z:
        if pd.isna(val):
            # No z-score available (e.g. warm-up period), stay flat
            position = 0
        
        elif position == 0:
            # Look for entry
            if entry_z < val < stop_z:
                # Short position
                position = -1
            elif -stop_z < val < -entry_z:
                # Long position
                position = 1
        elif position == 1:
            # In a long position, hold unless stop loss or mean reversion
            if val > -exit_z:
                # Trigger an exit
                position = 0
            elif val <  -stop_z:
                # Stop loss
                position = 0
            # Else we stay long
        elif position == -1:
            # In a short positon, hold unless stop loss or mean reversion
            if val < exit_z: 
                # Trigger an exit
                position = 0
            elif val > stop_z:
                position = 0
            # Else we stay short

        positions_list.append(position)
    
    return pd.Series(positions_list, index=z.index)

def rolling_zscore_walkforward(
    price_y: pd.Series,
    price_x: pd.Series,
    betas: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Rolling z-score of a spread under a time-varying hedge ratio.

    For each block of bars sharing the same beta, the spread is recomputed
    over the whole history using THAT beta, and the rolling mean/std are
    taken from that internally consistent series. This prevents a beta
    change from shifting the spread level underneath the rolling window,
    which would otherwise inject an artificial jump into the z-score at
    every refit date.

    Returns a Series of z-scores, NaN where beta is undefined (warm-up)
    or where the rolling window is incomplete.
    """
    z = pd.Series(np.nan, index=price_y.index)

    valid = betas.notna()
    if not valid.any():
        return z

    b = betas[valid]
    # Group consecutive bars that share the same beta into blocks
    block_id = (b != b.shift(1)).cumsum()

    for _, block in b.groupby(block_id):
        beta_val = block.iloc[0]
        # Whole-history spread at this single beta
        s = price_y - beta_val * price_x
        z_block = (s - s.rolling(window).mean()) / s.rolling(window).std()
        z.loc[block.index] = z_block.loc[block.index]

    return z


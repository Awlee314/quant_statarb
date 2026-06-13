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
    # By default pandas only uses up to window -1 bars

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
    "hold position between entry and exit thresholds" behavior — that
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



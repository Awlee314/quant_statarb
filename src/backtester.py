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

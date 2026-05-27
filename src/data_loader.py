import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE_DIR = _PROJECT_ROOT / 'data'/ 'cache'


def download_prices(tickers: list[str], start: str,
    end: str, cache_dir: str | Path = _DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """
    Download adjusted close prices for the specified tickers.
    - Use yfinance.
    - Cache each ticker as parquet at {cache_dir}/{ticker}.parquet
    - If cached and the date range is covered, load from cache.
    - We cache as parquet since reading from a parquet to DataFrame is much faster than from a csv.
    - Return a wide DataFrame: index=date, columns=tickers, values=adj close.
    - Forward-fill no more than 1 day for missing values; drop tickers with >5% missing.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    series_list = []

    for symbol in tickers:

        cache_path = Path(cache_dir) / f'{symbol}.parquet'

        stock = yf.Ticker(symbol)
        data = None
        if cache_path.exists():
            # If data already cached
            cached = pd.read_parquet(cache_path)
            # check if cached range covers correct dates
            if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
                # Only get rows from start to end
                data = cached.loc[start:end]
        # Cache miss
        if data is None:
                try:
                    data = stock.history(start=start,end=end, auto_adjust=True) # ensure adjusted prices
                    if data.empty:
                        print(f"[warn] no data for {symbol}, skipping to next ticker.")
                        continue
                    data.index = data.index.tz_localize(None) # strip tz
                    # Cache the data as a parquet
                    data.to_parquet(cache_path)
                except Exception as e:
                    print(f"[warn] failed to download {symbol}: {e}")
                    continue


        # Load the adjusted close data for each ticker into a series list
        series_list.append(data['Close'].rename(symbol))

    # Combine the series into a wide DataFrame
    all_data = pd.concat(series_list, axis=1)
    # Forward fill only 1 time max
    all_data = all_data.ffill(limit=1)

    # If >5% missing drop that ticker
    missing_pct = all_data.isna().mean()
    # Keep only the rows and then indices where less than 5% 
    keep = missing_pct[missing_pct <= 0.05].index
    drop = missing_pct[missing_pct > 0.05].index.to_list() # to list to make below if work nicely
    if drop:
        print(f"[info] dropped tickers with >5% missing values: {drop}")
    # Select only tickers which are keepers
    all_data = all_data[keep]
                
    return all_data




    

    

def compute_returns(prices: pd.DataFrame,method: str = "log"
) -> pd.DataFrame:
    """
    Compute returns. method='log' or 'simple'.
    Drop the first NaN row.
    """
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        # shift(1) moves all the data down by 1 to align for P_t/P_t-1.
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Invalid method: {method}. Use 'simple' or 'log'.")
    
    # Drops rows where every column is NaN (only the first row)
    return returns.dropna(how='all')


def get_universe(name: str) -> list[str]:
    """
    Return predefined ticker lists.
    name='banks' -> ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS']
    name='energy' -> ['XOM', 'CVX', 'COP', 'OXY', 'SLB', 'EOG']
    name='etf_pairs' -> ['SPY', 'IVV', 'QQQ', 'XLK']
    name='all' -> union of above
    """
    banks = ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS']
    energy = ['XOM', 'CVX', 'COP', 'OXY', 'SLB', 'EOG']
    etf_pairs = ['SPY', 'IVV', 'QQQ', 'XLK']

    if name == 'banks':
        return banks
    elif name == 'energy':
        return energy
    elif name == 'etf_pairs':
        return etf_pairs
    elif name == 'all':
        return banks + energy + etf_pairs
    else:
        raise ValueError(f"Unrecognized universe {name}. Options are 'banks', 'energy', 'etf_pairs', or 'all'.")



if __name__ == "__main__":
    
    tickers = ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'XOM', 
               'CVX', 'COP', 'OXY', 'SLB', 'EOG','SPY', 'IVV']
    # JP Morgan (JPM), Bank of America Corp (BAC), Citigroup Inc (C),
    #  Wells Fargo & Co (WFC), Goldman Sachs Group Inc (GS), Morgan Stanley (MS),
    # Exxon Mobil Corp (XOM), Chevron Corp (CVX), ConocoPhillips (COP),
    # Occidental Petroleum Corp (OXY), Slb NV (SLB), EOG Resources Inc (EOG), 
    # State Street SPDR S&P 500 ETF Trust (SPY), iShares Core S&P 500 ETF (IVV)
    # Invesco QQQ Trust, Series 1 (QQQ), State Street Technology Select Sector SPDR ETF (XLK)

    download_prices(tickers,'2020-01-01','2020-12-31')
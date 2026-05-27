from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

def adf_test(series: pd.Series, regression: str='c',
             autolag: str='AIC', sig_level: float = 0.05) -> dict:
    """
    Run an Augmented Dickey-Fuller test on a series.

    Returns a dictionary with
    - statistic: the ADF test statistic
    - p_value: p-value of the test
    - n_lags: number of lags used
    - n_obs: number of observations
    - critical_values: dictionary mapping percents to critical values
    - is_stationary: bool, true if p_value < 0.05

    Drops NaN values before tests.

    Null hypothesis is that the time series is non-stationary.
    This means that if the p-value is less than our significance level
    we can reject the null hypothesis and conclude that the time series
    is stationary.
    """

    results = adfuller(series.dropna(), regression=regression, autolag=autolag)

    dictionary = {
        "statistic": results[0],
        "p_value": results[1],
        "n_lags": results[2],
        "n_obs": results[3],
        "critical_values": results[4],
        "is_stationary": results[1] < sig_level
    }

    return dictionary

def ols_hedge_ratio(y: pd.Series, x: pd.Series) -> dict:
    """
    Estimate the hedge ratio via OLS: y = alpha + beta * x + u.
    
    Returns a dictionary with:
      - alpha: intercept
      - beta: slope (hedge ratio)
      - residuals: pd.Series, same index as inputs, with u_hat values
      - r_squared: coefficient of determination
    
    Drops rows where either input is NaN.
    """
    
    # Concatenate both series along columns and drop NaN rows
    df = pd.concat([y,x], axis=1).dropna()
    y_aligned = df.iloc[:,0]
    x_aligned = df.iloc[:,1]

    # Get the mean of x,y
    y_mean = y_aligned.mean()
    x_mean = x_aligned.mean()
    # Subtract x_mean and y_mean from the columns
    y_diff = y_aligned - y_mean
    x_diff = x_aligned - x_mean

    # Calculate beta as cov(x,y)/var(x)
    beta = (x_diff * y_diff).sum() / (x_diff**2).sum()
    alpha = y_mean - beta*x_mean
    # u = y - alpha - beta*x
    y_hat = alpha + beta*x_aligned
    residuals = y_aligned - y_hat
    ssr = ((residuals)**2).sum()
    tss = ((y_diff)**2).sum()
    r_2 = 1 - ssr/tss

    dictionary = {
        "alpha" : alpha,
        "beta" : beta,
        "residuals" : residuals,
        "r_squared" : r_2
    }
    
    return dictionary


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    sig_level: float = 0.05,
) -> dict:
    """
    Run the two-step Engle-Granger cointegration test.
    
    Step 1: OLS regress y on x.
    Step 2: ADF test on residuals.
    
    Returns dictionary with:
      - alpha: from OLS step
      - beta: hedge ratio
      - r_squared: from OLS step
      - residuals: pd.Series
      - adf_statistic: from ADF on residuals
      - adf_p_value: from ADF on residuals  (standard ADF p-value, not Engle-Granger)
      - is_cointegrated: bool, adf_p_value < sig_level
    """
    ols_dict = ols_hedge_ratio(y,x)
    adf_dict = adf_test(ols_dict["residuals"])

    dictionary = {
        'alpha' : ols_dict['alpha'],
        'beta' : ols_dict['beta'],
        'r_squared' : ols_dict['r_squared'],
        'residuals' : ols_dict['residuals'],
        'adf_statistic' : adf_dict['statistic'],
        'adf_p_value' : adf_dict['p_value'],
        'is_cointegrated' : adf_dict['p_value'] < sig_level

    }
    
    return dictionary

def half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion of a spread.
    
    Regresses Δu_t on u_{t-1}:
        Δu_t = λ * u_{t-1} + ε_t
    Then half_life = ln(0.5) / ln(1 + λ) = -ln(2) / ln(1 + λ).
    
    Returns half-life in periods (e.g., days if the spread is daily).
    Returns np.inf if the spread does not mean-revert (λ >= 0).
    Returns np.nan if there's insufficient data.
    
    Drops NaN values before computing.

    Since λ = ϕ - 1 we have ϕ = 1 + λ
    so when |ϕ| < 1 we require |1 + λ| < 1 for mean reversion.
    Thus λ < 0 and λ > -2 so -2 < λ < 0.
    """
    spread = spread.dropna()

    # u_lag is the series shifted down by 1 (u_t-1)
    u_lag = spread.shift(1)

    # compute the difference
    delta_u = (spread - u_lag).dropna()

    # regress delta_u on u_lag to get the beta (slope) which is lambda
    regress = ols_hedge_ratio(delta_u, u_lag)

    lamb = regress['beta']

    if lamb <= -2 or lamb >= 0:
        return np.inf

    half = - np.log(2) / np.log(1+lamb)

    if np.isnan(half):
        return np.inf
    
    return half

def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent via variance scaling.
    
    For lags τ = 2, 3, ..., max_lag:
        v(τ) = Var(X_{t+τ} - X_t)
    Then 2H = slope of log(v(τ)) vs log(τ).
    
    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    
    Returns np.nan if there's insufficient data.
    """
    # Drop NaNs and make as a numpy array
    arr = series.dropna().values
    variances = []

    for lag in range(1,max_lag+1):
        # Variance for each lag
        diff = arr[lag:] - arr[:-lag]
        # ddof = 1 to use n-1 not n for variance
        variances.append(np.var(diff,ddof=1))
    
    log_lags = np.log(np.arange(2,max_lag))
    log_vars = np.log(variances)
    res = ols_hedge_ratio(pd.Series(log_vars), pd.Series(log_lags))
    H = res['beta'] / 2
    return H

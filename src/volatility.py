import pandas as pd
import numpy as np
from scipy import stats

def ewma_volatility(returns, lam=0.94, freq=252):
    """
    EWMA (Exponentially Weighted Moving Average) volatility.
    RiskMetrics standard: lambda = 0.94 for daily data.

    Formula: σ²_t = λ × σ²_{t-1} + (1-λ) × r²_{t-1}

    More weight to recent returns — reacts quickly to market shocks.
    Unlike simple historical vol, EWMA vol rises sharply after a
    large move and gradually decays back down.

    Parameters
    ----------
    returns : Series of daily returns
    lam     : decay factor (default 0.94 = RiskMetrics standard)
    freq    : periods per year for annualisation (default 252)

    Returns
    -------
    Series of annualised EWMA volatility
    """
    r   = returns.dropna()
    n   = len(r)
    var = np.zeros(n)

    # Initialise with first squared return
    var[0] = r.iloc[0] ** 2

    # Recursive EWMA update
    for t in range(1, n):
        var[t] = lam * var[t-1] + (1 - lam) * r.iloc[t-1] ** 2

    vol = pd.Series(
        np.sqrt(var) * np.sqrt(freq),
        index=r.index,
        name=f"EWMA_Vol_λ{lam}"
    )
    return vol

def garch_volatility(returns, freq=252):
    """
    GARCH(1,1) conditional volatility.

    Formula: σ²_t = ω + α × r²_{t-1} + β × σ²_{t-1}

    Parameters estimated via Maximum Likelihood Estimation (MLE).

    Captures two key stylised facts of financial returns:
    1. Volatility clustering — calm periods followed by volatile periods
    2. Mean reversion — volatility reverts to long-run average

    Parameters
    ----------
    returns : Series of daily returns
    freq    : periods per year (default 252)

    Returns
    -------
    ann_vol : Series of annualised GARCH conditional volatility
    params  : dict of estimated parameters {omega, alpha, beta}
    result  : full arch model result object
    """
    try:
        from arch import arch_model

        # arch library works better with percentage returns
        r      = returns.dropna() * 100
        model  = arch_model(r, vol='Garch', p=1, q=1, dist='normal')
        result = model.fit(disp='off')

        # Conditional volatility (daily, in decimal)
        cond_vol = result.conditional_volatility / 100
        ann_vol  = cond_vol * np.sqrt(freq)
        ann_vol.name = "GARCH_Vol"

        # Key parameters
        omega = result.params['omega']
        alpha = result.params['alpha[1]']
        beta  = result.params['beta[1]']

        # Long-run variance = omega / (1 - alpha - beta)
        long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.nan
        long_run_vol = np.sqrt(long_run_var * 252 / 10000)  # annualised, decimal

        params = {
            "omega (ω)"           : round(omega, 6),
            "alpha (α) — ARCH"    : round(alpha, 4),
            "beta (β) — GARCH"    : round(beta, 4),
            "persistence (α+β)"   : round(alpha + beta, 4),
            "long-run annual vol" : round(long_run_vol * 100, 2),
            "log-likelihood"      : round(result.loglikelihood, 2),
            "AIC"                 : round(result.aic, 2),
        }

        return ann_vol, params, result

    except ImportError:
        print("Install arch library: pip install arch")
        return None, None, None

    except Exception as e:
        print(f"GARCH estimation failed: {e}")
        return None, None, None

def var_ewma(returns, confidence=0.95, lam=0.94):
    """
    Dynamic VaR using EWMA volatility.

    At each point in time uses the current EWMA vol estimate
    to compute VaR — gives a time-varying VaR that reacts to
    recent market conditions.

    Parameters
    ----------
    returns    : Series of daily returns
    confidence : VaR confidence level (default 0.95)
    lam        : EWMA decay factor (default 0.94)

    Returns
    -------
    Series of daily EWMA VaR estimates (positive = loss)
    """
    r        = returns.dropna()
    ewma_vol = ewma_volatility(r, lam=lam, freq=1)  # daily vol (not annualised)
    mu       = r.mean()
    z        = stats.norm.ppf(1 - confidence)
    var      = -(mu + z * ewma_vol)
    var.name = f"EWMA_VaR_{int(confidence*100)}"
    return var

def var_garch(returns, confidence=0.95, freq=252):
    """
    Dynamic VaR using GARCH(1,1) conditional volatility.

    Most sophisticated VaR — accounts for:
    1. Time-varying volatility
    2. Volatility clustering
    3. Mean reversion to long-run volatility

    Parameters
    ----------
    returns    : Series of daily returns
    confidence : VaR confidence level (default 0.95)
    freq       : periods per year (default 252)

    Returns
    -------
    var    : Series of daily GARCH VaR estimates (positive = loss)
    params : dict of GARCH parameters
    """
    ann_vol, params, result = garch_volatility(returns, freq=freq)

    if ann_vol is None:
        return None, None

    daily_vol = ann_vol / np.sqrt(freq)
    mu        = returns.mean()
    z         = stats.norm.ppf(1 - confidence)
    var       = -(mu + z * daily_vol)
    var.name  = f"GARCH_VaR_{int(confidence*100)}"
    return var, params

def compare_volatility_models(returns, freq=252):
    """
    Compare all three volatility models:
    1. Simple historical (constant)
    2. EWMA (time-varying, reactive)
    3. GARCH (time-varying, mean-reverting)

    Returns
    -------
    dict with vol series and summary statistics
    """
    r = returns.dropna()

    # Simple historical
    simple_vol = r.std() * np.sqrt(freq)

    # EWMA
    ewma_vol = ewma_volatility(r, lam=0.94, freq=freq)

    # GARCH
    garch_vol, garch_params, _ = garch_volatility(r, freq=freq)

    # Summary
    print(f"\n{'='*55}")
    print("  VOLATILITY MODEL COMPARISON")
    print(f"{'='*55}")
    print(f"  Simple Historical Vol : {simple_vol*100:.2f}% (constant)")

    if ewma_vol is not None:
        print(f"  EWMA Vol (current)    : {ewma_vol.iloc[-1]*100:.2f}%")
        print(f"  EWMA Vol (avg)        : {ewma_vol.mean()*100:.2f}%")
        print(f"  EWMA Vol (min)        : {ewma_vol.min()*100:.2f}%")
        print(f"  EWMA Vol (max)        : {ewma_vol.max()*100:.2f}%")

    if garch_vol is not None:
        print(f"\n  GARCH(1,1) Parameters:")
        for k, v in garch_params.items():
            print(f"    {k:<30} {v}")
        print(f"\n  GARCH Vol (current)   : {garch_vol.iloc[-1]*100:.2f}%")
        print(f"  GARCH Vol (avg)       : {garch_vol.mean()*100:.2f}%")
        print(f"  GARCH Vol (min)       : {garch_vol.min()*100:.2f}%")
        print(f"  GARCH Vol (max)       : {garch_vol.max()*100:.2f}%")

    print(f"{'='*55}")

    return {
        "simple"     : simple_vol,
        "ewma"       : ewma_vol,
        "garch"      : garch_vol,
        "garch_params": garch_params,
    }

def print_var_comparison(returns, confidence=0.95):
    """
    Compare VaR estimates across all three volatility models.
    Shows current VaR — most recent estimate for each model.
    """
    from risk import var_historical, var_parametric

    r = returns.dropna()

    # Point-in-time VaR estimates
    hist_var  = var_historical(r, confidence)
    param_var = var_parametric(r, confidence)
    ewma_var  = var_ewma(r, confidence).iloc[-1]
    garch_var_series, _ = var_garch(r, confidence)
    garch_var = garch_var_series.iloc[-1] if garch_var_series is not None else np.nan

    print(f"\n{'='*55}")
    print(f"  VaR COMPARISON ({int(confidence*100)}% confidence, 1-day)")
    print(f"{'='*55}")
    print(f"  Historical VaR    : {hist_var*100:.3f}%  (full sample)")
    print(f"  Parametric VaR    : {param_var*100:.3f}%  (normal dist)")
    print(f"  EWMA VaR          : {ewma_var*100:.3f}%  (λ=0.94, current)")
    print(f"  GARCH VaR         : {garch_var*100:.3f}%  (current)")
    print(f"{'='*55}")
    print(f"  Interpretation:")
    if ewma_var > hist_var:
        print(f"  → EWMA > Historical: market MORE volatile than average")
    else:
        print(f"  → EWMA < Historical: market LESS volatile than average")
    if garch_var > param_var:
        print(f"  → GARCH > Parametric: fat tails detected")
    print(f"{'='*55}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    returns = pd.read_csv(
        "data/processed/returns_daily.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    print(f"Loaded: {len(returns)} daily returns")

    # Compare all volatility models
    vol_models = compare_volatility_models(returns)

    # Compare VaR estimates
    print_var_comparison(returns, confidence=0.95)
    print_var_comparison(returns, confidence=0.99)

    # Save volatility series
    import os
    os.makedirs("data/processed", exist_ok=True)
    if vol_models["ewma"] is not None:
        vol_models["ewma"].to_csv("data/processed/ewma_vol.csv")
    if vol_models["garch"] is not None:
        vol_models["garch"].to_csv("data/processed/garch_vol.csv")
    print("\nSaved volatility series to data/processed/")
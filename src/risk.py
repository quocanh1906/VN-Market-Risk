import pandas as pd
import numpy as np
from scipy import stats

# ── Value at Risk ───────────────────────────────────────────────────────────────

def var_historical(returns, confidence=0.95, horizon=1):
    """
    Historical simulation VaR.

    Uses the empirical distribution of past returns — no distributional
    assumption. The most widely used method in practice.

    Parameters
    ----------
    returns    : Series of returns
    confidence : confidence level (default 0.95 = 95%)
    horizon    : holding period in days (default 1)

    Returns
    -------
    float : VaR as a positive number (loss)
    """
    r = returns.dropna()
    # Scale to horizon using square root of time rule
    scaled = r * np.sqrt(horizon)
    var = -np.percentile(scaled, (1 - confidence) * 100)
    return var

def var_parametric(returns, confidence=0.95, horizon=1):
    """
    Parametric (variance-covariance) VaR.

    Assumes returns are normally distributed. Fast to compute but
    underestimates tail risk since real returns have fat tails.

    Parameters
    ----------
    returns    : Series of returns
    confidence : confidence level (default 0.95)
    horizon    : holding period in days (default 1)

    Returns
    -------
    float : VaR as a positive number (loss)
    """
    r   = returns.dropna()
    mu  = r.mean() * horizon
    sig = r.std() * np.sqrt(horizon)
    var = -(mu + stats.norm.ppf(1 - confidence) * sig)
    return var

def var_monte_carlo(returns, confidence=0.95, horizon=1,
                    n_simulations=10000, seed=42):
    """
    Monte Carlo VaR.

    Simulates future return paths using the historical mean and
    volatility. More flexible than parametric but still assumes
    normality unless we use a fat-tailed distribution.

    Parameters
    ----------
    returns      : Series of returns
    confidence   : confidence level (default 0.95)
    horizon      : holding period in days (default 1)
    n_simulations: number of simulated paths (default 10,000)
    seed         : random seed for reproducibility

    Returns
    -------
    float : VaR as a positive number (loss)
    """
    np.random.seed(seed)
    r   = returns.dropna()
    mu  = r.mean()
    sig = r.std()

    # Simulate horizon-day returns
    simulated = np.random.normal(mu, sig, (n_simulations, horizon))
    path_returns = simulated.sum(axis=1)  # sum daily returns over horizon

    var = -np.percentile(path_returns, (1 - confidence) * 100)
    return var

# ── Expected Shortfall (CVaR) ───────────────────────────────────────────────────

def cvar_historical(returns, confidence=0.95, horizon=1):
    """
    Historical simulation CVaR (Expected Shortfall).

    Average loss in the worst (1-confidence)% of scenarios.
    More informative than VaR — tells you the expected loss
    GIVEN that you're in the tail.

    CVaR is the standard risk measure for Basel III/IV bank regulation.
    """
    r      = returns.dropna()
    scaled = r * np.sqrt(horizon)
    var    = var_historical(r, confidence, horizon)
    cvar   = -scaled[scaled < -var].mean()
    return cvar

def cvar_parametric(returns, confidence=0.95, horizon=1):
    """
    Parametric CVaR under normality assumption.

    Analytical formula: CVaR = mu + sig * phi(z) / (1-confidence)
    where phi is the standard normal PDF and z is the confidence quantile.
    """
    r   = returns.dropna()
    mu  = r.mean() * horizon
    sig = r.std() * np.sqrt(horizon)
    z   = stats.norm.ppf(1 - confidence)
    cvar = -(mu - sig * stats.norm.pdf(z) / (1 - confidence))
    return cvar

# ── Rolling Risk Metrics ────────────────────────────────────────────────────────

def rolling_var(returns, window=252, confidence=0.95, horizon=1):
    """
    Compute rolling historical VaR over a sliding window.

    Shows how risk evolves over time — spikes during crises,
    falls during calm periods.

    Parameters
    ----------
    window : rolling window in days (default 252 = 1 year)
    """
    var_series = returns.rolling(window).apply(
        lambda x: var_historical(pd.Series(x), confidence, horizon),
        raw=False
    )
    var_series.name = f"VaR_{int(confidence*100)}%_{horizon}d"
    return var_series

def rolling_volatility(returns, window=21):
    """
    Rolling annualised volatility.
    window=21 ≈ 1 month of trading days.
    """
    vol = returns.rolling(window).std() * np.sqrt(252)
    vol.name = f"Vol_{window}d"
    return vol

# ── Drawdown Analysis ───────────────────────────────────────────────────────────

def compute_drawdowns(returns):
    """
    Compute full drawdown series and key drawdown statistics.

    Returns
    -------
    drawdown    : Series of drawdown at each point in time
    stats       : dict of max drawdown, avg drawdown, recovery stats
    """
    cum         = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max

    # Find drawdown periods
    is_drawdown = drawdown < 0
    max_dd      = drawdown.min()

    # Find worst drawdown period
    end_idx   = drawdown.idxmin()
    start_idx = cum[:end_idx].idxmax()

    # Recovery — first time we reach new high after max drawdown
    post_dd     = cum[end_idx:]
    peak_value  = cum[start_idx]
    recovery    = post_dd[post_dd >= peak_value]
    recovery_date = recovery.index[0] if len(recovery) > 0 else None

    dd_stats = {
        "Max Drawdown (%)"       : round(max_dd * 100, 2),
        "Drawdown Start"         : start_idx.date(),
        "Drawdown End"           : end_idx.date(),
        "Recovery Date"          : recovery_date.date() if recovery_date else "Not yet",
        "Drawdown Duration (days)": (end_idx - start_idx).days,
        "Recovery Duration (days)": (recovery_date - end_idx).days if recovery_date else "N/A",
        "Avg Drawdown (%)"       : round(drawdown[is_drawdown].mean() * 100, 2),
        "% Time in Drawdown"     : round(is_drawdown.mean() * 100, 2),
    }

    return drawdown, dd_stats

# ── Full Risk Report ────────────────────────────────────────────────────────────

def compute_risk_report(returns, name="Portfolio", freq=252):
    """
    Compute comprehensive risk report for a return series.

    Parameters
    ----------
    returns : Series of returns
    name    : portfolio name for display
    freq    : periods per year (252=daily, 52=weekly, 12=monthly)
    """
    r = returns.dropna()

    # Basic stats
    ann_return = r.mean() * freq
    ann_vol    = r.std() * np.sqrt(freq)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan
    skewness   = r.skew()
    kurtosis   = r.kurtosis()  # excess kurtosis

    # VaR at multiple confidence levels
    var_95_hist  = var_historical(r, 0.95)
    var_99_hist  = var_historical(r, 0.99)
    var_95_param = var_parametric(r, 0.95)
    var_99_param = var_parametric(r, 0.99)
    var_95_mc    = var_monte_carlo(r, 0.95)

    # CVaR
    cvar_95_hist  = cvar_historical(r, 0.95)
    cvar_99_hist  = cvar_historical(r, 0.99)
    cvar_95_param = cvar_parametric(r, 0.95)

    # Drawdown
    drawdown, dd_stats = compute_drawdowns(r)

    # Normality test
    _, pvalue = stats.jarque_bera(r)

    report = {
        "Name"                    : name,
        "Observations"            : len(r),
        "Period"                  : f"{r.index[0].date()} to {r.index[-1].date()}",
        ""                        : "─" * 40,
        "Ann. Return (%)"         : round(ann_return * 100, 2),
        "Ann. Volatility (%)"     : round(ann_vol * 100, 2),
        "Sharpe Ratio"            : round(sharpe, 3),
        "Skewness"                : round(skewness, 3),
        "Excess Kurtosis"         : round(kurtosis, 3),
        "Jarque-Bera p-value"     : round(pvalue, 4),
        " "                       : "─" * 40,
        "VaR 95% Historical (%)"  : round(var_95_hist * 100, 3),
        "VaR 99% Historical (%)"  : round(var_99_hist * 100, 3),
        "VaR 95% Parametric (%)"  : round(var_95_param * 100, 3),
        "VaR 99% Parametric (%)"  : round(var_99_param * 100, 3),
        "VaR 95% Monte Carlo (%)" : round(var_95_mc * 100, 3),
        "  "                      : "─" * 40,
        "CVaR 95% Historical (%)": round(cvar_95_hist * 100, 3),
        "CVaR 99% Historical (%)": round(cvar_99_hist * 100, 3),
        "CVaR 95% Parametric (%)": round(cvar_95_param * 100, 3),
    }

    # Add drawdown stats
    report.update({"   ": "─" * 40})
    report.update(dd_stats)

    return report, drawdown

def print_risk_report(report):
    """Print formatted risk report."""
    print(f"\n{'='*55}")
    print(f"  RISK REPORT: {report['Name']}")
    print(f"{'='*55}")
    for k, v in report.items():
        if k == "Name":
            continue
        if str(v).startswith("─"):
            print(f"  {v}")
        else:
            print(f"  {k:<35} {v}")
    print(f"{'='*55}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Load daily returns
    returns = pd.read_csv(
        "data/processed/returns_daily.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    print(f"Loaded: {len(returns)} daily returns")

    # Compute full risk report
    report, drawdown = compute_risk_report(returns, "E1VFVN30 ETF", freq=252)
    print_risk_report(report)
import pandas as pd
import numpy as np

# ── Historical Stress Scenarios ─────────────────────────────────────────────────
# Key market crisis periods relevant to Vietnamese equity market

STRESS_SCENARIOS = {
    "COVID Crash"         : ("2020-01-20", "2020-03-30"),
    "VN Market Crash 2022": ("2022-04-01", "2022-11-30"),
    "Trade War 2018"      : ("2018-04-01", "2018-10-31"),
    "China Selloff 2015"  : ("2015-06-01", "2015-09-30"),
    "Rising Rates 2021-22": ("2021-11-01", "2022-06-30"),
    "Global GFC 2008"     : ("2008-01-01", "2009-03-31"),
}

def run_historical_stress(returns, scenarios=None):
    """
    Run historical stress tests — measure actual performance
    during known crisis periods.

    Parameters
    ----------
    returns   : Series of daily returns
    scenarios : dict of {name: (start, end)} — uses STRESS_SCENARIOS if None

    Returns
    -------
    DataFrame with performance metrics for each scenario
    """
    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    results = []

    for name, (start, end) in scenarios.items():
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        # Filter returns to scenario window
        mask    = (returns.index >= start_ts) & (returns.index <= end_ts)
        r_period = returns[mask]

        if len(r_period) < 5:
            print(f"  ⚠ {name}: insufficient data ({len(r_period)} days)")
            continue

        # Compute metrics
        cum_return   = (1 + r_period).prod() - 1
        ann_vol      = r_period.std() * np.sqrt(252)
        max_daily_loss = r_period.min()
        worst_week   = r_period.rolling(5).sum().min()
        n_days       = len(r_period)

        # Drawdown within scenario
        cum          = (1 + r_period).cumprod()
        drawdown     = (cum / cum.cummax() - 1)
        max_dd       = drawdown.min()

        results.append({
            "Scenario"           : name,
            "Period"             : f"{start} to {end}",
            "Trading Days"       : n_days,
            "Total Return (%)"   : round(cum_return * 100, 2),
            "Ann. Volatility (%)": round(ann_vol * 100, 2),
            "Max Drawdown (%)"   : round(max_dd * 100, 2),
            "Worst Day (%)"      : round(max_daily_loss * 100, 2),
            "Worst Week (%)"     : round(worst_week * 100, 2),
        })

    return pd.DataFrame(results).set_index("Scenario")

def run_hypothetical_stress(returns, scenarios=None):
    """
    Run hypothetical stress tests — apply custom shocks to
    the current portfolio.

    Parameters
    ----------
    returns   : Series of returns (used for volatility scaling)
    scenarios : dict of {name: shock_pct} — instantaneous shocks

    Returns
    -------
    DataFrame showing portfolio loss under each scenario
    """
    if scenarios is None:
        scenarios = {
            "Market -10%"         : -0.10,
            "Market -20%"         : -0.20,
            "Market -30%"         : -0.30,
            "Market -40%"         : -0.40,
            "2022 VN Crash Repeat": -0.47,
            "COVID Repeat"        : -0.35,
            "GFC Repeat"          : -0.60,
        }

    r   = returns.dropna()
    vol = r.std() * np.sqrt(252)

    results = []
    for name, shock in scenarios.items():
        # Simple: portfolio loss = shock (assumes beta = 1 to market)
        # More sophisticated: could adjust by beta
        results.append({
            "Scenario"           : name,
            "Market Shock (%)"   : round(shock * 100, 1),
            "Portfolio Loss (%)": round(shock * 100, 1),
            "Days at Current Vol": round(abs(shock) / (vol / np.sqrt(252)), 1),
        })

    return pd.DataFrame(results).set_index("Scenario")

def var_stressed(returns, confidence=0.95, window=252):
    """
    Stressed VaR — compute VaR using only the worst historical period.

    Basel III requires banks to compute VaR over a stressed period
    (typically the worst 252-day window in the historical data).
    This gives a more conservative risk estimate.

    Returns
    -------
    tuple: (stressed_var, stressed_period_start, stressed_period_end)
    """
    r = returns.dropna()

    # Find the 252-day window with highest volatility
    rolling_vol = r.rolling(window).std()
    worst_start_idx = rolling_vol.idxmax()

    # Get that window
    loc          = r.index.get_loc(worst_start_idx)
    start_loc    = max(0, loc - window + 1)
    stressed_r   = r.iloc[start_loc:loc + 1]

    stressed_var = -np.percentile(stressed_r, (1 - confidence) * 100)
    period_start = stressed_r.index[0].date()
    period_end   = stressed_r.index[-1].date()

    return stressed_var, period_start, period_end

def print_stress_results(hist_df, hyp_df, stressed_var, period):
    """Print formatted stress test results."""

    print(f"\n{'='*65}")
    print("  HISTORICAL STRESS TESTS")
    print(f"{'='*65}")
    print(hist_df.to_string())

    print(f"\n{'='*65}")
    print("  HYPOTHETICAL STRESS TESTS")
    print(f"{'='*65}")
    print(hyp_df.to_string())

    print(f"\n{'='*65}")
    print("  STRESSED VaR (Basel III approach)")
    print(f"{'='*65}")
    print(f"  Stressed period : {period[0]} to {period[1]}")
    print(f"  Stressed VaR 95%: {stressed_var*100:.3f}%")
    print(f"  (vs normal VaR 95% — compare to risk report)")
    print(f"{'='*65}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    returns = pd.read_csv(
        "data/processed/returns_daily.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    print(f"Loaded: {len(returns)} daily returns")

    # Historical stress tests
    print("\nRunning historical stress tests...")
    hist_results = run_historical_stress(returns)

    # Hypothetical stress tests
    print("Running hypothetical stress tests...")
    hyp_results = run_hypothetical_stress(returns)

    # Stressed VaR
    print("Computing stressed VaR...")
    s_var, s_start, s_end = var_stressed(returns)

    # Print results
    print_stress_results(
        hist_results, hyp_results,
        s_var, (s_start, s_end)
    )

    # Save
    import os
    os.makedirs("data/processed", exist_ok=True)
    hist_results.to_csv("data/processed/stress_historical.csv")
    hyp_results.to_csv("data/processed/stress_hypothetical.csv")
    print("\nSaved to data/processed/")
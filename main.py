import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, "src")

from data import download_etf, build_returns
from risk import (compute_risk_report, print_risk_report,
                  rolling_var, rolling_volatility,
                  compute_drawdowns)
from stress import (run_historical_stress, run_hypothetical_stress,
                    print_stress_results, var_stressed)
from volatility import (compare_volatility_models, print_var_comparison,
                        ewma_volatility, garch_volatility,
                        var_ewma, var_garch)
from performance import (plot_risk_dashboard, plot_var_comparison,
                         plot_stress_results, plot_volatility_comparison)

print("=" * 60)
print("VN Market Risk — Full Pipeline")
print("E1VFVN30 ETF (2014-2024)")
print("=" * 60)

os.makedirs("output", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── 1. Load data ────────────────────────────────────────────────────────────────
print("\nStep 1: Loading data...")

returns_path = "data/processed/returns_daily.csv"

if os.path.exists(returns_path):
    returns = pd.read_csv(returns_path, index_col=0,
                          parse_dates=True).squeeze()
    print(f"  Loaded from cache: {len(returns)} daily returns")
else:
    print("  Downloading from vnstock...")
    prices        = download_etf()
    returns, _, _ = build_returns(prices)
    returns.to_csv(returns_path)
    print(f"  Downloaded: {len(returns)} daily returns")

print(f"  Period: {returns.index[0].date()} to {returns.index[-1].date()}")

# ── 2. Risk report ──────────────────────────────────────────────────────────────
print("\nStep 2: Computing risk metrics...")
report, drawdown = compute_risk_report(returns, "E1VFVN30 ETF", freq=252)
print_risk_report(report)

# ── 3. Rolling metrics ──────────────────────────────────────────────────────────
print("\nStep 3: Computing rolling metrics...")
r_var = rolling_var(returns, window=252, confidence=0.95)
r_vol = rolling_volatility(returns, window=21)

# Stressed VaR
s_var, s_start, s_end = var_stressed(returns)
print(f"  Rolling VaR computed  : {len(r_var.dropna())} obs")
print(f"  Rolling Vol computed  : {len(r_vol.dropna())} obs")
print(f"  Stressed VaR 95%      : {s_var*100:.3f}%")
print(f"  Stressed period       : {s_start} to {s_end}")

# ── 4. Volatility models ────────────────────────────────────────────────────────
print("\nStep 4: Fitting volatility models...")

# EWMA
print("  Fitting EWMA (λ=0.94)...")
ewma_vol = ewma_volatility(returns, lam=0.94, freq=252)

# GARCH
print("  Fitting GARCH(1,1)...")
garch_vol, garch_params, _ = garch_volatility(returns, freq=252)

# Compare all models
vol_models = compare_volatility_models(returns)

# Compare VaR across methods
print_var_comparison(returns, confidence=0.95)
print_var_comparison(returns, confidence=0.99)

# Save volatility series
ewma_vol.to_csv("data/processed/ewma_vol.csv")
if garch_vol is not None:
    garch_vol.to_csv("data/processed/garch_vol.csv")

# ── 5. Stress tests ─────────────────────────────────────────────────────────────
print("\nStep 5: Running stress tests...")
hist_results = run_historical_stress(returns)
hyp_results  = run_hypothetical_stress(returns)
print_stress_results(hist_results, hyp_results, s_var, (s_start, s_end))

# Save
hist_results.to_csv("output/stress_historical.csv")
hyp_results.to_csv("output/stress_hypothetical.csv")

# Save risk report
report_df = pd.DataFrame(
    [(k, v) for k, v in report.items()
     if not str(v).startswith("─")],
    columns=["Metric", "Value"]
).set_index("Metric")
report_df.to_csv("output/risk_report.csv")

# ── 6. Plots ────────────────────────────────────────────────────────────────────
print("\nStep 6: Generating charts...")

# Risk dashboard
plot_risk_dashboard(returns, drawdown, r_var, r_vol)

# VaR method comparison
plot_var_comparison(returns)

# Stress test results
plot_stress_results(hist_results)

# Volatility model comparison — NEW
plot_volatility_comparison(returns, ewma_vol, garch_vol)

# ── 7. Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Ann. Return          : {report['Ann. Return (%)']:>8}%")
print(f"  Ann. Volatility      : {report['Ann. Volatility (%)']:>8}%")
print(f"  Sharpe Ratio         : {report['Sharpe Ratio']:>8}")
print(f"  VaR 95% Historical   : {report['VaR 95% Historical (%)']:>8}%")
print(f"  CVaR 95% Historical  : {report['CVaR 95% Historical (%)']:>8}%")
print(f"  EWMA Vol (current)   : {round(ewma_vol.iloc[-1]*100, 2):>8}%")
if garch_vol is not None:
    print(f"  GARCH Vol (current)  : {round(garch_vol.iloc[-1]*100, 2):>8}%")
    print(f"  GARCH persistence    : {garch_params['persistence (α+β)']:>8}")
print(f"  Stressed VaR 95%     : {round(s_var*100, 3):>8}%")
print(f"  Max Drawdown         : {report['Max Drawdown (%)']:>8}%")
print(f"  Output files         : output/")
print("=" * 60)
print("\nDone!")
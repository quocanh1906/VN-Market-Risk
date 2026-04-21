import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import os

def plot_risk_dashboard(returns, drawdown, rolling_var, rolling_vol):
    """
    Plot comprehensive risk dashboard with 4 panels:
    1. Cumulative return
    2. Rolling volatility
    3. Rolling VaR
    4. Drawdown
    """
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("VN30 ETF (E1VFVN30) — Market Risk Dashboard (2014-2024)",
                 fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.4)

    # ── Panel 1: Cumulative Return ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cum = (1 + returns).cumprod()
    ax1.plot(cum.index, cum.values, color="steelblue", linewidth=1.5)
    ax1.axhline(y=1, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Cumulative Return", fontsize=10)
    ax1.grid(alpha=0.3)

    # Shade crisis periods
    crises = [
        ("2015-06-01", "2015-09-30", "China Selloff"),
        ("2018-04-01", "2018-10-31", "Trade War"),
        ("2020-01-20", "2020-03-30", "COVID"),
        ("2022-04-01", "2022-11-30", "VN Crash"),
    ]
    for start, end, label in crises:
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.15, color="red")
        mid = pd.Timestamp(start) + (pd.Timestamp(end) -
                                      pd.Timestamp(start)) / 2
        ax1.text(mid, ax1.get_ylim()[1] * 0.95, label,
                 ha="center", fontsize=7, color="darkred")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 2: Rolling Volatility ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(rolling_vol.index, rolling_vol.values * 100,
             color="darkorange", linewidth=1.2)
    ax2.axhline(y=rolling_vol.mean() * 100, color="grey",
                linestyle="--", linewidth=0.8,
                label=f"Mean: {rolling_vol.mean()*100:.1f}%")
    ax2.set_ylabel("Volatility (%)")
    ax2.set_title("Rolling 21-day Annualised Volatility", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 3: Rolling VaR ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(rolling_var.index, rolling_var.values * 100,
             color="red", linewidth=1.2, label="Rolling VaR 95%")
    ax3.axhline(y=rolling_var.mean() * 100, color="grey",
                linestyle="--", linewidth=0.8,
                label=f"Mean: {rolling_var.mean()*100:.2f}%")
    ax3.set_ylabel("VaR (%)")
    ax3.set_title("Rolling 252-day Historical VaR (95%, 1-day)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Panel 4: Drawdown ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(drawdown.index, drawdown.values * 100, 0,
                     color="steelblue", alpha=0.5)
    ax4.axhline(y=drawdown.mean() * 100, color="grey",
                linestyle="--", linewidth=0.8,
                label=f"Avg: {drawdown.mean()*100:.1f}%")
    ax4.set_ylabel("Drawdown (%)")
    ax4.set_title("Drawdown from Peak", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.savefig("output/risk_dashboard.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("Saved to output/risk_dashboard.png")

def plot_var_comparison(returns):
    """
    Compare VaR methods — Historical vs Parametric vs Monte Carlo.
    Also shows return distribution with normal overlay.
    """
    from risk import (var_historical, var_parametric, var_monte_carlo,
                      cvar_historical, cvar_parametric)
    from scipy import stats

    r = returns.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VaR Method Comparison — E1VFVN30",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Return distribution ───────────────────────────────────────────
    ax = axes[0]
    ax.hist(r * 100, bins=80, density=True,
            color="steelblue", alpha=0.6, label="Actual returns")

    # Normal distribution overlay
    x = np.linspace(r.min() * 100, r.max() * 100, 200)
    ax.plot(x, stats.norm.pdf(x, r.mean() * 100, r.std() * 100),
            color="red", linewidth=2, label="Normal distribution")

    # VaR lines
    var_h = var_historical(r, 0.95)
    var_p = var_parametric(r, 0.95)
    ax.axvline(x=-var_h * 100, color="darkblue", linestyle="--",
               linewidth=1.5, label=f"VaR 95% Hist: {var_h*100:.2f}%")
    ax.axvline(x=-var_p * 100, color="darkred", linestyle="--",
               linewidth=1.5, label=f"VaR 95% Param: {var_p*100:.2f}%")

    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Return Distribution vs Normal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2: VaR method comparison bar chart ────────────────────────────────
    ax = axes[1]

    methods   = ["Historical\n95%", "Parametric\n95%", "Monte Carlo\n95%",
                 "Historical\n99%", "Parametric\n99%"]
    var_vals  = [
        var_historical(r, 0.95) * 100,
        var_parametric(r, 0.95) * 100,
        var_monte_carlo(r, 0.95) * 100,
        var_historical(r, 0.99) * 100,
        var_parametric(r, 0.99) * 100,
    ]
    cvar_vals = [
        cvar_historical(r, 0.95) * 100,
        cvar_parametric(r, 0.95) * 100,
        np.nan,
        cvar_historical(r, 0.99) * 100,
        cvar_parametric(r, 0.99) * 100,
    ]

    x     = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, var_vals, width,
                   label="VaR", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2,
                   [c if not np.isnan(c) else 0 for c in cvar_vals],
                   width, label="CVaR", color="darkorange", alpha=0.8)

    ax.set_xlabel("Method & Confidence Level")
    ax.set_ylabel("Risk (%)")
    ax.set_title("VaR vs CVaR by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("output/var_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("Saved to output/var_comparison.png")

def plot_stress_results(hist_df):
    """
    Visualise historical stress test results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Historical Stress Test Results — E1VFVN30",
                 fontsize=13, fontweight="bold")

    colors = ["red" if x < -20 else "orange" if x < -10
              else "steelblue" for x in hist_df["Total Return (%)"]]

    # Total return by scenario
    axes[0].barh(hist_df.index, hist_df["Total Return (%)"],
                 color=colors, alpha=0.8)
    axes[0].axvline(x=0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Total Return (%)")
    axes[0].set_title("Total Return During Crisis")
    axes[0].grid(alpha=0.3, axis="x")
    for i, v in enumerate(hist_df["Total Return (%)"]):
        axes[0].text(v - 0.5 if v < 0 else v + 0.5, i,
                     f"{v:.1f}%", va="center", fontsize=9,
                     ha="right" if v < 0 else "left")

    # Max drawdown by scenario
    dd_colors = ["red" if x < -30 else "orange" if x < -15
                 else "steelblue" for x in hist_df["Max Drawdown (%)"]]
    axes[1].barh(hist_df.index, hist_df["Max Drawdown (%)"],
                 color=dd_colors, alpha=0.8)
    axes[1].axvline(x=0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Max Drawdown (%)")
    axes[1].set_title("Max Drawdown During Crisis")
    axes[1].grid(alpha=0.3, axis="x")
    for i, v in enumerate(hist_df["Max Drawdown (%)"]):
        axes[1].text(v - 0.5, i, f"{v:.1f}%",
                     va="center", ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig("output/stress_results.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("Saved to output/stress_results.png")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from risk import (compute_risk_report, rolling_var,
                      rolling_volatility, compute_drawdowns)

    returns = pd.read_csv(
        "data/processed/returns_daily.csv",
        index_col=0, parse_dates=True
    ).squeeze()

    hist_stress = pd.read_csv(
        "data/processed/stress_historical.csv",
        index_col=0
    )

    # Compute rolling metrics
    r_var = rolling_var(returns, window=252, confidence=0.95)
    r_vol = rolling_volatility(returns, window=21)
    drawdown, _ = compute_drawdowns(returns)

    # Plots
    os.makedirs("output", exist_ok=True)
    plot_risk_dashboard(returns, drawdown, r_var, r_vol)
    plot_var_comparison(returns)
    plot_stress_results(hist_stress)
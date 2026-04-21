import pandas as pd
import numpy as np
import os

def download_etf(symbol="E1VFVN30", start="2014-01-01", end="2024-12-31"):
    """
    Download daily ETF price history from vnstock.
    Returns a Series of daily close prices.
    """
    try:
        from vnstock import Quote
        quote = Quote(symbol=symbol, source='KBS')
        df    = quote.history(start=start, end=end, interval='1D')

        if df is None or df.empty:
            return None

        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')['close']
        df = df.sort_index()
        df.name = symbol
        print(f"✓ {symbol}: {len(df)} days "
              f"({df.index[0].date()} to {df.index[-1].date()})")
        return df

    except Exception as e:
        print(f"✗ {symbol}: {e}")
        return None

def build_returns(prices):
    """
    Compute daily, weekly and monthly returns from price series.
    """
    daily   = prices.pct_change().dropna()
    weekly  = prices.resample('W-FRI').last().pct_change().dropna()
    monthly = prices.resample('ME').last().pct_change().dropna()

    daily.name   = "daily"
    weekly.name  = "weekly"
    monthly.name = "monthly"

    return daily, weekly, monthly

def build_price_matrix(symbol="E1VFVN30",
                        start="2014-01-01",
                        end="2024-12-31"):
    """
    Download ETF prices and compute returns at all frequencies.
    Saves to data/processed/.
    """
    print("=" * 60)
    print(f"VN Market Risk — {symbol}")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)

    # Download
    print(f"\nDownloading {symbol}...")
    prices = download_etf(symbol=symbol, start=start, end=end)

    if prices is None:
        raise ValueError(f"Failed to download {symbol}")

    # Compute returns
    daily, weekly, monthly = build_returns(prices)

    print(f"\nReturn series:")
    print(f"  Daily   : {len(daily)} obs "
          f"({daily.index[0].date()} to {daily.index[-1].date()})")
    print(f"  Weekly  : {len(weekly)} obs")
    print(f"  Monthly : {len(monthly)} obs")

    # Basic stats
    print(f"\nDaily return summary:")
    print(f"  Mean    : {daily.mean()*252*100:.2f}% ann.")
    print(f"  Std     : {daily.std()*252**0.5*100:.2f}% ann.")
    print(f"  Min     : {daily.min()*100:.2f}%")
    print(f"  Max     : {daily.max()*100:.2f}%")

    # Save
    prices.to_csv("data/processed/prices.csv")
    daily.to_csv("data/processed/returns_daily.csv")
    weekly.to_csv("data/processed/returns_weekly.csv")
    monthly.to_csv("data/processed/returns_monthly.csv")
    print(f"\nSaved to data/processed/")

    return prices, daily, weekly, monthly

if __name__ == "__main__":
    prices, daily, weekly, monthly = build_price_matrix()
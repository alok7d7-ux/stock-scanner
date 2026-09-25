import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. CORE TECHNICAL INDICATORS
# ==========================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0):
    """Calculates Supertrend line and Direction (-1: Bullish, 1: Bearish)."""
    atr = calculate_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    st_dir = 1
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
            continue
            
        c_close = df['Close'].iloc[i]
        p_close = df['Close'].iloc[i-1]
        c_upper = upper_band.iloc[i]
        c_lower = lower_band.iloc[i]
        
        c_upper = c_upper if (c_upper < upper_band.iloc[i-1] or p_close > upper_band.iloc[i-1]) else upper_band.iloc[i-1]
        c_lower = c_lower if (c_lower > lower_band.iloc[i-1] or p_close < lower_band.iloc[i-1]) else lower_band.iloc[i-1]
        
        if st_dir == 1:
            st_dir = -1 if c_close > c_upper else 1
        else:
            st_dir = 1 if c_close < c_lower else -1
            
        supertrend.iloc[i] = c_lower if st_dir == -1 else c_upper
        direction.iloc[i] = st_dir
        
    return supertrend, direction

def calculate_nto(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculates Normalized Trend Oscillator (NTO)."""
    momentum = df['Close'].diff(length)
    highest_high = df['High'].rolling(window=length).max()
    lowest_low = df['Low'].rolling(window=length).min()
    range_val = highest_high - lowest_low
    nto = np.where(range_val != 0, (momentum / range_val) * 100, 0)
    return pd.Series(nto, index=df.index)

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates Volume Weighted Average Price (VWAP)."""
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).cumsum() / v.cumsum()

# ==========================================
# 2. NSE 200 TICKER LOADER
# ==========================================

def get_nse200_tickers() -> list:
    """Fetches the NIFTY 200 stock list dynamically or falls back to major constituents."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
    try:
        # NSE blocks basic Python user agents, so we pass browser headers
        headers = {'User-Agent': 'Mozilla/5.0'}
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        df_nse = pd.read_csv(urllib.request.urlopen(req))
        symbols = [f"{sym}.NS" for sym in df_nse['Symbol'].dropna().unique()]
        print(f"Loaded {len(symbols)} tickers from official NIFTY 200 CSV.")
        return symbols
    except Exception as e:
        print(f"Could not download live NSE 200 CSV ({e}). Falling back to primary NSE watchlist...")
        # Fallback list of major NIFTY 200 constituents
        fallback = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
            "INFY.NS", "ITC.NS", "SBIN.NS", "LTIM.NS", "LT.NS", "HINDUNILVR.NS",
            "AXISBANK.NS", "KOTAKBANK.NS", "HCLTECH.NS", "ADANIENT.NS", "SUNPHARMA.NS",
            "TATAMOTORS.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TITAN.NS",
            "ULTRACEMCO.NS", "BAJFINANCE.NS", "M&M.NS", "MARUTI.NS", "TATASTEEL.NS",
            "COALINDIA.NS", "JSWSTEEL.NS", "ASIANPAINT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS",
            "GRASIM.NS", "BPCL.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
            "EICHERMOT.NS", "CIPLA.NS", "SBILIFE.NS", "DIVISLAB.NS", "BRITANNIA.NS",
            "TATACONSUM.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS", "WIPRO.NS", "BAJAJ-AUTO.NS",
            "NESTLEIND.NS", "HINDALCO.NS", "BEL.NS", "HAL.NS", "TRENT.NS", "ZOMATO.NS"
        ]
        return fallback

# ==========================================
# 3. SCANNING LOGIC FOR INDIVIDUAL TICKER
# ==========================================

def scan_symbol(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2mo", interval="1d")
    
    if df.empty or len(df) < 21:
        return None
    
    # Dual Supertrend Calculations
    _, st1_dir = calculate_supertrend(df, period=10, multiplier=2.0)
    _, st2_dir = calculate_supertrend(df, period=21, multiplier=1.0)
    dual_st_bullish = (st1_dir.iloc[-1] == -1) and (st2_dir.iloc[-1] == -1)
    dual_st_bearish = (st1_dir.iloc[-1] == 1) and (st2_dir.iloc[-1] == 1)
    
    # NTO Indicator
    nto = calculate_nto(df, length=14)
    nto_bullish = nto.iloc[-1] > 80.0
    nto_bearish = nto.iloc[-1] < -80.0
    
    # Market Structure Breakouts
    highest_high = df['High'].iloc[-10:-1].max()
    lowest_low = df['Low'].iloc[-10:-1].min()
    mss_bullish = df['Close'].iloc[-1] > highest_high
    mss_bearish = df['Close'].iloc[-1] < lowest_low
    
    # VWAP & Volume Spike
    vwap = calculate_vwap(df)
    vol_sma = df['Volume'].rolling(window=20).mean()
    above_vwap = df['Close'].iloc[-1] > vwap.iloc[-1]
    vol_spike = df['Volume'].iloc[-1] > vol_sma.iloc[-1]
    
    # Scoring Engine
    score = 0
    score += 2 if above_vwap else -2
    score += 2 if dual_st_bullish else (-2 if dual_st_bearish else 0)
    score += 2 if nto_bullish else (-2 if nto_bearish else 0)
    score += 2 if mss_bullish else (-2 if mss_bearish else 0)
    score += (1 if df['Close'].iloc[-1] >= df['Open'].iloc[-1] else -1) if vol_spike else 0
    
    # Signal Assignment
    if score >= 4:
        action = "STRONG BUY"
    elif score >= 2:
        action = "BUY"
    elif score <= -4:
        action = "STRONG SELL"
    elif score <= -2:
        action = "SELL"
    else:
        action = "WAIT / NEUTRAL"
        
    return {
        "Symbol": symbol.replace(".NS", ""),
        "Close": round(df['Close'].iloc[-1], 2),
        "Action": action,
        "Score": score,
        "Dual ST": "BULL" if dual_st_bullish else ("BEAR" if dual_st_bearish else "MIXED"),
        "NTO (>80)": nto_bullish,
        "Above VWAP": above_vwap,
        "Vol Spike": vol_spike
    }

# ==========================================
# 4. RUNNER & BATCH PROCESSOR
# ==========================================

if __name__ == "__main__":
    watchlist = get_nse200_tickers()
    
    results = []
    print(f"Scanning {len(watchlist)} NSE stocks... Please wait.\n")
    
    for i, sym in enumerate(watchlist, 1):
        try:
            res = scan_symbol(sym)
            if res:
                results.append(res)
            print(f"[{i}/{len(watchlist)}] Processed: {sym}", end="\r")
        except Exception as e:
            continue
            
    df_results = pd.DataFrame(results)
    
    # Filter for actionable signals
    buy_signals = df_results[df_results['Action'].isin(["STRONG BUY", "BUY"])].sort_values(by="Score", ascending=False)
    sell_signals = df_results[df_results['Action'].isin(["STRONG SELL", "SELL"])].sort_values(by="Score", ascending=True)
    
    print("\n\n================ TOP BULLISH STOCKS (BUY) ================")
    print(buy_signals.to_string(index=False) if not buy_signals.empty else "No Strong Buy signals today.")
    
    print("\n================ TOP BEARISH STOCKS (SELL) ================")
    print(sell_signals.to_string(index=False) if not sell_signals.empty else "No Strong Sell signals today.")

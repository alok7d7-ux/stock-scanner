import urllib.request
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
# 2. NSE F&O TICKER LOADER
# ==========================================

def get_nifty_fno_tickers() -> list:
    """Fetches official NSE F&O stock list or falls back to major derivatives watchlist."""
    url = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        df_fno = pd.read_csv(urllib.request.urlopen(req))
        # Filter out index symbols (NIFTY, BANKNIFTY, etc.)
        symbols = [f"{sym.strip()}.NS" for sym in df_fno['UNDERLYING'].dropna().unique() if sym.strip() not in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']]
        print(f"Loaded {len(symbols)} F&O stock tickers from NSE.")
        return symbols
    except Exception as e:
        print(f"Could not fetch dynamic F&O list ({e}). Using primary F&O fallback watchlist...")
        fallback = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
            "INFY.NS", "ITC.NS", "SBIN.NS", "LTIM.NS", "LT.NS", "HINDUNILVR.NS",
            "AXISBANK.NS", "KOTAKBANK.NS", "HCLTECH.NS", "ADANIENT.NS", "SUNPHARMA.NS",
            "TATAMOTORS.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TITAN.NS",
            "ULTRACEMCO.NS", "BAJFINANCE.NS", "M&M.NS", "MARUTI.NS", "TATASTEEL.NS",
            "COALINDIA.NS", "JSWSTEEL.NS", "ASIANPAINT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS",
            "DIVISLAB.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "DRREDDY.NS", "CIPLA.NS", "MCX.NS"
        ]
        return fallback

# ==========================================
# 3. SCANNING & COC CALCULATION LOGIC
# ==========================================

def scan_symbol(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2mo", interval="1d")
    
    if df.empty or len(df) < 21:
        return None
    
    raw_symbol = symbol.replace(".NS", "")
    tv_link = f"https://www.tradingview.com/chart/?symbol=NSE:{raw_symbol}"
    
    # 1. Technical Indicators
    _, st1_dir = calculate_supertrend(df, period=10, multiplier=2.0)
    _, st2_dir = calculate_supertrend(df, period=21, multiplier=1.0)
    dual_st_bullish = (st1_dir.iloc[-1] == -1) and (st2_dir.iloc[-1] == -1)
    dual_st_bearish = (st1_dir.iloc[-1] == 1) and (st2_dir.iloc[-1] == 1)
    
    nto = calculate_nto(df, length=14)
    nto_bullish = nto.iloc[-1] > 80.0
    nto_bearish = nto.iloc[-1] < -80.0
    
    highest_high = df['High'].iloc[-10:-1].max()
    lowest_low = df['Low'].iloc[-10:-1].min()
    mss_bullish = df['Close'].iloc[-1] > highest_high
    mss_bearish = df['Close'].iloc[-1] < lowest_low
    
    vwap = calculate_vwap(df)
    vol_sma = df['Volume'].rolling(window=20).mean()
    above_vwap = df['Close'].iloc[-1] > vwap.iloc[-1]
    vol_spike = df['Volume'].iloc[-1] > vol_sma.iloc[-1]
    
    # 2. Cost of Carry (CoC) Proxy Calculation
    # Compares current Close vs 5-day SMA representing price premium expansion/shrinkage
    price_sma5 = df['Close'].rolling(window=5).mean().iloc[-1]
    coc_pct = ((df['Close'].iloc[-1] - price_sma5) / price_sma5) * 100
    
    if coc_pct > 0.1:
        coc_status = "POSITIVE (Expand)"
        coc_score = 2
    elif coc_pct < -0.1:
        coc_status = "NEGATIVE (Shrink)"
        coc_score = -2
    else:
        coc_status = "FLAT / NEUTRAL"
        coc_score = 0
        
    # 3. Multi-Factor Scoring Engine
    score = 0
    score += 2 if above_vwap else -2
    score += 2 if dual_st_bullish else (-2 if dual_st_bearish else 0)
    score += 2 if nto_bullish else (-2 if nto_bearish else 0)
    score += 2 if mss_bullish else (-2 if mss_bearish else 0)
    score += (1 if df['Close'].iloc[-1] >= df['Open'].iloc[-1] else -1) if vol_spike else 0
    score += coc_score  # Incorporate CoC weight into total score
    
    # Signal Assignment
    if score >= 5:
        action = "STRONG BUY"
    elif score >= 2:
        action = "BUY"
    elif score <= -5:
        action = "STRONG SELL"
    elif score <= -2:
        action = "SELL"
    else:
        action = "WAIT / NEUTRAL"
        
    return {
        "Symbol": raw_symbol,
        "Close": round(df['Close'].iloc[-1], 2),
        "Action": action,
        "Score": score,
        "CoC": coc_status,
        "CoC_Val": round(coc_pct, 2),
        "TV_Link": tv_link
    }

# ==========================================
# 4. RUNNER & BATCH PROCESSOR
# ==========================================

if __name__ == "__main__":
    watchlist = get_nifty_fno_tickers()
    
    results = []
    print(f"Scanning {len(watchlist)} NSE F&O stocks... Please wait.\n")
    
    for i, sym in enumerate(watchlist, 1):
        try:
            res = scan_symbol(sym)
            if res:
                results.append(res)
            print(f"[{i}/{len(watchlist)}] Processed: {sym}", end="\r")
        except Exception:
            continue
            
    df_results = pd.DataFrame(results)
    
    buy_signals = df_results[df_results['Action'].isin(["STRONG BUY", "BUY"])].sort_values(by="Score", ascending=False)
    sell_signals = df_results[df_results['Action'].isin(["STRONG SELL", "SELL"])].sort_values(by="Score", ascending=True)
    
    print("\n\n================ TOP BULLISH F&O STOCKS (BUY) ================")
    if not buy_signals.empty:
        for _, row in buy_signals.iterrows():
            print(f"• {row['Symbol']} | Close: ₹{row['Close']} | Score: {row['Score']} | CoC: {row['CoC']} ({row['CoC_Val']}%) | Action: {row['Action']}")
            print(f"  Chart Link: {row['TV_Link']}\n")
    else:
        print("No Strong Buy signals in F&O stocks today.")
    
    print("================ TOP BEARISH F&O STOCKS (SELL) ================")
    if not sell_signals.empty:
        for _, row in sell_signals.iterrows():
            print(f"• {row['Symbol']} | Close: ₹{row['Close']} | Score: {row['Score']} | CoC: {row['CoC']} ({row['CoC_Val']}%) | Action: {row['Action']}")
            print(f"  Chart Link: {row['TV_Link']}\n")
    else:
        print("No Strong Sell signals in F&O stocks today.")

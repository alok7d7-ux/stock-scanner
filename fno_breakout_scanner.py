import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. CORE TECHNICAL INDICATORS
# ==========================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.0):
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

# ==========================================
# 2. NSE F&O WATCHLIST (~180 STOCKS)
# ==========================================

def get_fno_watchlist() -> list:
    fno_list = [
        "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT",
        "ADANIPORTS", "ALKEM", "AMBUJACEMENT", "ANGELONE", "APOLLOHOSP", "APOLLOTYRE", 
        "ASHOKLEY", "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", 
        "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", 
        "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG", "BHARTIARTL", "BHEL", 
        "BIOCON", "BPCL", "BRITANNIA", "BSOFT", "CANBK", "CANFINHOME", "CHAMBLFERT", 
        "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", 
        "CROMPTON", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DIVISLAB", "DIXON", 
        "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", 
        "GLENMARK", "GMRINFRA", "GNFC", "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD", 
        "HAL", "HAVELLS", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", 
        "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", 
        "IDEA", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM", "INDIAMART", "INDIGO", 
        "INDUSINDBK", "INDUSTOWER", "INFY", "IOC", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", 
        "JKCEMENT", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LALPATHLAB", "LT", "LTIM", 
        "LTF", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", 
        "MCDOWELL-N", "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", 
        "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", 
        "OBEROIRLTY", "OFSS", "ONGC", "PAGEIND", "PERSISTENT", "PETRONET", "PFC", 
        "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POWERGRID", "PVRINOX", "RAMCOCEM", 
        "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", 
        "SHRIRAMFIN", "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM", 
        "TATACOMM", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", 
        "TITAN", "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", 
        "UPL", "VEDL", "VOLTAS", "WIPRO", "ZEEL", "ZYDUSLIFE"
    ]
    return [f"{sym}.NS" for sym in fno_list]

# ==========================================
# 3. 20-DAY BREAKOUT SCANNER ENGINE
# ==========================================

def scan_20d_breakout(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="3mo", interval="1d")
    
    if df.empty or len(df) < 21:
        return None
    
    raw_symbol = symbol.replace(".NS", "")
    tv_link = f"https://www.tradingview.com/chart/?symbol=NSE:{raw_symbol}"
    
    # 1. 20-Day High & Low Check (Excluding current candle)
    high_20d = df['High'].iloc[-21:-1].max()
    low_20d = df['Low'].iloc[-21:-1].min()
    
    curr_close = df['Close'].iloc[-1]
    curr_high = df['High'].iloc[-1]
    curr_low = df['Low'].iloc[-1]
    
    is_20d_high = curr_high >= high_20d
    is_20d_low = curr_low <= low_20d
    
    # 2. Volume Spike Check
    vol_sma = df['Volume'].rolling(window=20).mean()
    curr_vol = df['Volume'].iloc[-1]
    vol_spike = curr_vol > (vol_sma.iloc[-1] * 1.5)
    
    # 3. Cost of Carry (CoC) Expansion / Shrinkage
    price_sma5 = df['Close'].rolling(window=5).mean().iloc[-1]
    coc_pct = ((curr_close - price_sma5) / price_sma5) * 100
    
    # 4. Supertrend Alignment
    _, st1_dir = calculate_supertrend(df, period=10, multiplier=2.0)
    _, st2_dir = calculate_supertrend(df, period=21, multiplier=1.0)
    dual_st_bull = (st1_dir.iloc[-1] == -1) and (st2_dir.iloc[-1] == -1)
    dual_st_bear = (st1_dir.iloc[-1] == 1) and (st2_dir.iloc[-1] == 1)
    
    # Ranking Score Model
    score = 0
    setup_type = "NEUTRAL"
    option_action = "NONE"
    
    if is_20d_high:
        score += 3
        setup_type = "20-DAY HIGH BREAKOUT (HOD)"
        option_action = "BUY DEEP OTM CALL"
    elif is_20d_low:
        score -= 3
        setup_type = "20-DAY LOW BREAKOUT (LOD)"
        option_action = "BUY DEEP OTM PUT"
        
    score += (2 if vol_spike else 0) if is_20d_high else (-2 if vol_spike else 0)
    score += (2 if coc_pct > 0.1 else 0) if is_20d_high else (-2 if coc_pct < -0.1 else 0)
    score += (2 if dual_st_bull else 0) if is_20d_high else (-2 if dual_st_bear else 0)
    
    if setup_type == "NEUTRAL":
        return None
        
    return {
        "Symbol": raw_symbol,
        "Close": round(curr_close, 2),
        "Setup": setup_type,
        "Option_Target": option_action,
        "Score": abs(score),
        "Raw_Score": score,
        "CoC_%": round(coc_pct, 2),
        "Vol_Spike": "YES" if vol_spike else "NO",
        "TV_Link": tv_link
    }

# ==========================================
# 4. EXECUTION RUNNER & RANKING
# ==========================================

if __name__ == "__main__":
    watchlist = get_fno_watchlist()
    print(f"Scanning {len(watchlist)} NSE F&O stocks for 20-Day High/Low Breakouts...\n")
    
    results = []
    for i, sym in enumerate(watchlist, 1):
        try:
            res = scan_20d_breakout(sym)
            if res:
                results.append(res)
            print(f"[{i}/{len(watchlist)}] Processed: {sym}", end="\r")
        except Exception:
            continue
            
    df_res = pd.DataFrame(results)
    
    if not df_res.empty:
        # Separate Call and Put Setups and Rank by Score
        call_setups = df_res[df_res['Raw_Score'] > 0].sort_values(by="Score", ascending=False)
        put_setups = df_res[df_res['Raw_Score'] < 0].sort_values(by="Score", ascending=False)
        
        print("\n\n================ 🚀 RANKED CALL SETUPS (20-DAY HIGH BREAKOUTS) ================")
        if not call_setups.empty:
            for rank, (_, row) in enumerate(call_setups.iterrows(), 1):
                print(f"Rank #{rank} | Stock: {row['Symbol']} | Close: ₹{row['Close']} | Score: {row['Score']}/9")
                print(f"  Setup: {row['Setup']} | Action: {row['Option_Target']} | Vol Spike: {row['Vol_Spike']} | CoC: {row['CoC_%']}%")
                print(f"  Chart Link: {row['TV_Link']}\n")
        else:
            print("No 20-Day High Breakout Call setups today.")
            
        print("================ 🔻 RANKED PUT SETUPS (20-DAY LOW BREAKOUTS) ================")
        if not put_setups.empty:
            for rank, (_, row) in enumerate(put_setups.iterrows(), 1):
                print(f"Rank #{rank} | Stock: {row['Symbol']} | Close: ₹{row['Close']} | Score: {row['Score']}/9")
                print(f"  Setup: {row['Setup']} | Action: {row['Option_Target']} | Vol Spike: {row['Vol_Spike']} | CoC: {row['CoC_%']}%")
                print(f"  Chart Link: {row['TV_Link']}\n")
        else:
            print("No 20-Day Low Breakout Put setups today.")
    else:
        print("No 20-Day High/Low breakouts detected in F&O stocks today.")

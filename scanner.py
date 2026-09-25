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
        except Exception:
            continue
            
    df_results = pd.DataFrame(results)
    
    buy_signals = df_results[df_results['Action'].isin(["STRONG BUY", "BUY"])].sort_values(by="Score", ascending=False)
    sell_signals = df_results[df_results['Action'].isin(["STRONG SELL", "SELL"])].sort_values(by="Score", ascending=True)
    
    print("\n\n================ TOP BULLISH STOCKS (BUY) ================")
    if not buy_signals.empty:
        for _, row in buy_signals.iterrows():
            print(f"[{row['Symbol']}] (₹{row['Close']} | Score: {row['Score']} | Action: {row['Action']})")
            print(f"--> https://www.tradingview.com/chart/?symbol=NSE:{row['Symbol']}\n")
    else:
        print("No Strong Buy signals today.")
    
    print("================ TOP BEARISH STOCKS (SELL) ================")
    if not sell_signals.empty:
        for _, row in sell_signals.iterrows():
            print(f"[{row['Symbol']}] (₹{row['Close']} | Score: {row['Score']} | Action: {row['Action']})")
            print(f"--> https://www.tradingview.com/chart/?symbol=NSE:{row['Symbol']}\n")
    else:
        print("No Strong Sell signals today.")

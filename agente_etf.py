import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# CONFIGURAZIONE
# ==========================================
TELEGRAM_TOKEN = "IL_TUO_TOKEN_TELEGRAM"
TELEGRAM_CHAT_ID = "IL_TUO_CHAT_ID"

PORTFOLIO = ["SPY", "QQQ", "TLT"]
WATCHLIST = ["URTH", "EEM", "VNQ", "GLD"]

# ==========================================
# MOTORE DI SCORING CENTRALIZZATO
# ==========================================
def analyze_df_engine(df):
    """
    Calcola gli indicatori tecnici e genera uno score da 0.0 a 1.0.
    Include il kill-switch ADX per fasi laterali.
    """
    if df.empty or len(df) < 35:
        return 0.0

    # Calcolo Indicatori Tecnici tramite pandas_ta
    df.ta.ema(length=10, append=True)
    df.ta.sma(length=31, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.adx(length=14, append=True)
    
    # Calcolo Heikin Ashi
    ha_df = ta.ha(df['Open'], df['High'], df['Low'], df['Close'])
    df = pd.concat([df, ha_df], axis=1)

    latest = df.iloc[-1]
    
    # 1. Kill-Switch ADX (Fase laterale)
    # Se l'ADX è inferiore a 20, il trend è debole, score azzerato.
    if pd.isna(latest.get('ADX_14')) or latest['ADX_14'] < 20:
        return 0.0

    score = 0.0
    
    # 2. Trend di fondo (EMA10 vs SMA31)
    if latest['EMA_10'] > latest['SMA_31']: score += 0.25
    
    # 3. Momentum (RSI)
    if 50 < latest['RSI_14'] < 70: score += 0.25
    
    # 4. MACD (Istogramma positivo)
    if latest['MACDh_12_26_9'] > 0: score += 0.25
    
    # 5. Price Action (Heikin Ashi verde e forte)
    if latest['HA_close'] > latest['HA_open']: score += 0.25

    return min(score, 1.0)

# ==========================================
# ANALISI MULTI-TIMEFRAME & RISCHIO
# ==========================================
def analyze_flash_ticker(ticker):
    """
    Scarica dati 1D e 1H, calcola gli score e definisce i livelli di rischio.
    """
    try:
        # Fetch dati 1D (6 mesi)
        df_1d = yf.download(ticker, period="6mo", interval="1d", progress=False)
        # Fetch dati 1H (1 mese)
        df_1h = yf.download(ticker, period="1mo", interval="1h", progress=False)

        if df_1d.empty or df_1h.empty:
            return None

        # Calcolo Score
        score_1d = analyze_df_engine(df_1d)
        score_1h = analyze_df_engine(df_1h)

        # Gestione Rischio (ATR Daily)
        df_1d.ta.atr(length=14, append=True)
        latest_1d = df_1d.iloc[-1]
        current_price = latest_1d['Close']
        prev_price = df_1d.iloc[-2]['Close']
        
        daily_pct_change = ((current_price - prev_price) / prev_price) * 100
        atr = latest_1d['ATR_14']
        
        # Setup base: Stop Loss a 1.5 ATR, Take Profit a 3 ATR (R:R 1:2)
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 3)
        sizing_risk = (atr * 1.5) / current_price * 100 # Rischio % per azione

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "pct_change": round(daily_pct_change, 2),
            "score_1d": round(score_1d, 2),
            "score_1h": round(score_1h, 2),
            "atr": round(atr, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "sizing_risk_pct": round(sizing_risk, 2)
        }
    except Exception as e:
        print(f"Errore su {ticker}: {e}")
        return None

# ==========================================
# GESTIONE MESSAGGI TELEGRAM
# ==========================================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, json=payload)

def format_telegram_alert(title, results):
    """
    Formatta gerarchicamente l'output per Telegram ordinato per Score Daily.
    """
    if not results:
        return f"<b>{title}</b>\nNessun segnale rilevante."

    # Ordinamento primario: Score 1D decrescente, poi Score 1H
    sorted_results = sorted(results, key=lambda x: (x['score_1d'], x['score_1h']), reverse=True)

    msg = f"<b>{title}</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    
    for r in sorted_results:
        trend_emoji = "🟢" if r['score_1d'] >= 0.75 else "🟡" if r['score_1d'] >= 0.5 else "🔴"
        change_sign = "+" if r['pct_change'] > 0 else ""
        
        msg += f"{trend_emoji} <b>{r['ticker']}</b> | {r['price']}$ ({change_sign}{r['pct_change']}%)\n"
        msg += f"   ├ Score 1D: {r['score_1d']} | Score 1H: {r['score_1h']}\n"
        msg += f"   ├ Rischio: SL {r['stop_loss']}$ | TP {r['take_profit']}$\n"
        msg += f"   └ ATR: {r['atr']} | Size Rischio: {r['sizing_risk_pct']}%\n\n"
        
    return msg

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print(f"[{datetime.now()}] Avvio Analisi FLASH...")
    
    # 1. Processamento Portfolio
    portfolio_results = []
    for ticker in PORTFOLIO:
        res = analyze_flash_ticker(ticker)
        if res: portfolio_results.append(res)
        
    # 2. Processamento Watchlist
    watchlist_results = []
    for ticker in WATCHLIST:
        res = analyze_flash_ticker(ticker)
        if res: watchlist_results.append(res)
        
    # 3. Invio Alert separati
    if portfolio_results:
        msg_portfolio = format_telegram_alert("📊 ANALISI FLASH: PORTFOLIO", portfolio_results)
        send_telegram_message(msg_portfolio)
        print("Alert Portfolio inviato.")
        
    if watchlist_results:
        msg_watchlist = format_telegram_alert("👀 ANALISI FLASH: WATCHLIST", watchlist_results)
        send_telegram_message(msg_watchlist)
        print("Alert Watchlist inviato.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agente di Trading - Analisi ETF
Motore di scoring con kill-switch ADX per fasi laterali.
"""

import os
from datetime import datetime

import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# ==========================================
# CONFIGURAZIONE
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

PORTFOLIO = ["SPY", "QQQ", "TLT"]
WATCHLIST = ["URTH", "EEM", "VNQ", "GLD"]


# ==========================================
# MOTORE DI SCORING CENTRALIZZATO
# ==========================================
def analyze_df_engine(df: pd.DataFrame) -> float:
    """
    Calcola gli indicatori tecnici e genera uno score da 0.0 a 1.0.
    Include il kill-switch ADX per fasi laterali.
    """
    if df.empty or len(df) < 35:
        return 0.0

    # Normalizza MultiIndex (yfinance a volte restituisce colonne annidate)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Lavora su una copia per non mutare il DataFrame originale
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

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

    # 1. Kill-Switch ADX (fase laterale → score azzerato)
    adx_val = latest.get('ADX_14')
    if pd.isna(adx_val) or adx_val < 20:
        return 0.0

    score = 0.0

    # 2. Trend di fondo (EMA10 vs SMA31)
    if latest['EMA_10'] > latest['SMA_31']:
        score += 0.25

    # 3. Momentum (RSI)
    if 50 < latest['RSI_14'] < 70:
        score += 0.25

    # 4. MACD (istogramma positivo)
    if latest['MACDh_12_26_9'] > 0:
        score += 0.25

    # 5. Price Action (Heikin Ashi verde)
    if latest['HA_close'] > latest['HA_open']:
        score += 0.25

    return min(score, 1.0)


# ==========================================
# ANALISI MULTI-TIMEFRAME & RISCHIO
# ==========================================
def analyze_flash_ticker(ticker: str):
    """
    Scarica dati 1D e 1H, calcola gli score e definisce i livelli di rischio.
    Ritorna un dict oppure None se i dati sono insufficienti.
    """
    try:
        df_1d = yf.download(ticker, period="6mo", interval="1d",
                            auto_adjust=True, progress=False)
        df_1h = yf.download(ticker, period="1mo", interval="1h",
                            auto_adjust=True, progress=False)

        if df_1d.empty or df_1h.empty:
            print(f"⚠️ {ticker}: dati insufficienti (1D o 1H vuoti)")
            return None

        # Normalizza MultiIndex
        if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = df_1d.columns.get_level_values(0)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)

        # Score 1D e 1H
        score_1d = analyze_df_engine(df_1d)
        score_1h = analyze_df_engine(df_1h)

        # --- Gestione Rischio (ATR Daily) ---
        df_atr = df_1d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_atr.ta.atr(length=14, append=True)

        latest_atr = df_atr.iloc[-1]
        current_price = float(df_1d['Close'].iloc[-1])
        prev_price = float(df_1d['Close'].iloc[-2])

        daily_pct_change = ((current_price - prev_price) / prev_price) * 100.0

        # pandas_ta usa il nome 'ATRr_14' (RMA-based), non 'ATR_14'
        atr = float(latest_atr['ATRr_14'])

        # Setup: SL 1.5 ATR, TP 3 ATR (R:R 1:2)
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 3.0)
        sizing_risk = (atr * 1.5) / current_price * 100.0

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "pct_change": round(daily_pct_change, 2),
            "score_1d": round(score_1d, 2),
            "score_1h": round(score_1h, 2),
            "atr": round(atr, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "sizing_risk_pct": round(sizing_risk, 2),
        }

    except Exception as e:
        print(f"❌ Errore su {ticker}: {e}")
        return None


# ==========================================
# GESTIONE MESSAGGI TELEGRAM
# ==========================================
def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti!")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"❌ Errore Telegram ({resp.status_code}): {resp.text}")
            return False
        print("✅ Messaggio Telegram inviato")
        return True
    except Exception as e:
        print(f"❌ Eccezione invio Telegram: {e}")
        return False


def format_telegram_alert(title: str, results: list) -> str:
    """Formatta gerarchicamente l'output per Telegram ordinato per Score Daily."""
    if not results:
        return f"<b>{title}</b>\nNessun segnale rilevante."

    # Ordinamento: Score 1D decrescente, poi Score 1H
    sorted_results = sorted(
        results,
        key=lambda x: (x['score_1d'], x['score_1h']),
        reverse=True,
    )

    msg = f"<b>{title}</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"

    for r in sorted_results:
        if r['score_1d'] >= 0.75:
            trend_emoji = "🟢"
        elif r['score_1d'] >= 0.5:
            trend_emoji = "🟡"
        else:
            trend_emoji = "🔴"

        change_sign = "+" if r['pct_change'] > 0 else ""

        msg += f"{trend_emoji} <b>{r['ticker']}</b> | {r['price']}$ ({change_sign}{r['pct_change']}%)\n"
        msg += f"   ├ Score 1D: {r['score_1d']} | Score 1H: {r['score_1h']}\n"
        msg += f"   ├ Rischio: SL {r['stop_loss']}$ | TP {r['take_profit']}$\n"
        msg += f"   └ ATR: {r['atr']} | Size Rischio: {r['sizing_risk_pct']}%\n\n"

    return msg


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("📊 AGENTE ETF - ANALISI MULTI-TIMEFRAME")
    print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)

    # 1. Portfolio
    portfolio_results = []
    print("\n💰 ANALISI PORTFOLIO ETF")
    for ticker in PORTFOLIO:
        print(f"   → {ticker}")
        res = analyze_flash_ticker(ticker)
        if res:
            portfolio_results.append(res)

    # 2. Watchlist
    watchlist_results = []
    print("\n👁️ ANALISI WATCHLIST ETF")
    for ticker in WATCHLIST:
        print(f"   → {ticker}")
        res = analyze_flash_ticker(ticker)
        if res:
            watchlist_results.append(res)

    # 3. Invio Telegram
    if portfolio_results:
        print(f"\n📩 Invio alert Portfolio ({len(portfolio_results)} ticker)...")
        msg_portfolio = format_telegram_alert("📊 ANALISI FLASH: PORTFOLIO ETF", portfolio_results)
        send_telegram_message(msg_portfolio)
    else:
        print("\n⚠️ Nessun risultato per il Portfolio, nessun invio.")

    if watchlist_results:
        print(f"\n📩 Invio alert Watchlist ({len(watchlist_results)} ticker)...")
        msg_watchlist = format_telegram_alert("👀 ANALISI FLASH: WATCHLIST ETF", watchlist_results)
        send_telegram_message(msg_watchlist)
    else:
        print("\n⚠️ Nessun risultato per la Watchlist, nessun invio.")

    print(f"\n🏁 Completato: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()

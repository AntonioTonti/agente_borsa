#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera
Invio: 13:00 e 18:00 UTC (14:00 e 19:00 IT)
"""

import os
import sys
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
from typing import List, Dict

# Configurazione
sys.path.append('.')
from config import (
    load_titoli_csv, DAILY_PERIOD, DAILY_INTERVAL, DAILY_MIN_POINTS
)

# ============================================================================
# INDICATORI GIORNALIERI (RAPIDI)
# ============================================================================

def analyze_daily_ticker(ticker: str) -> List[str]:
    """Analisi rapida giornaliera"""
    signals = []
    
    try:
        # Download dati
        df = yf.download(ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL, progress=False)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            return signals
        
        # Pulizia dati
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        volume = df['Volume']
        
        # 1. EMA10 vs MA31 (indicatore principale)
        if len(close) >= 32:
            import ta
            ema10 = ta.trend.ema_indicator(close, window=10)
            ma31 = ta.trend.sma_indicator(close, window=31)
            
            if len(ema10) > 1 and len(ma31) > 1:
                ema_now = float(ema10.iloc[-1])
                ma_now = float(ma31.iloc[-1])
                ema_prev = float(ema10.iloc[-2])
                ma_prev = float(ma31.iloc[-2])
                
                if ema_now > ma_now and ema_prev <= ma_prev:
                    signals.append("📈 EMA10 > MA31 (CROSSOVER UP)")
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append("📉 MA31 > EMA10 (CROSSOVER DOWN)")
                elif ema_now > ma_now:
                    signals.append("🟢 EMA10 sopra MA31")
                else:
                    signals.append("🔴 MA31 sopra EMA10")
        
        # 2. RSI giornaliero
        if len(close) >= 15:
            import ta
            rsi = ta.momentum.rsi(close, window=14)
            if len(rsi) > 0:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    signals.append("⚠️  RSI > 70 (IPERCOMPRATO)")
                elif rsi_val < 30:
                    signals.append("⚠️  RSI < 30 (IPERVENDUTO)")
        
        # 3. Volume vs media
        if len(volume) >= 10:
            avg_volume = float(volume.tail(10).mean())
            current_volume = float(volume.iloc[-1])
            if current_volume > avg_volume * 1.5:
                signals.append("📊 Volume +50%")
        
        return signals
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return []

# ============================================================================
# FUNZIONI DI OUTPUT
# ============================================================================

def format_daily_message(portfolio_signals: Dict, watchlist_signals: Dict, descriptions: Dict) -> str:
    """Formatta messaggio giornaliero"""
    time_str = datetime.now().strftime("%d/%m %H:%M")
    header = f"📈 *SEGNALI GIORNALIERI {time_str}*\n\n"
    
    parts = []
    
    if portfolio_signals:
        parts.append("💰 *PORTAFOGLIO ATTIVO*")
        for ticker, signals in portfolio_signals.items():
            desc = descriptions.get(ticker, ticker)
            parts.append(f"• *{ticker}* - {desc}")
            for signal in signals:
                parts.append(f"  {signal}")
        parts.append("")
    
    if watchlist_signals:
        parts.append("👁️  *WATCHLIST*")
        for ticker, signals in watchlist_signals.items():
            desc = descriptions.get(ticker, ticker)
            parts.append(f"• *{ticker}* - {desc}")
            for signal in signals:
                parts.append(f"  {signal}")
    
    if not portfolio_signals and not watchlist_signals:
        return header + "📭 Nessun segnale rilevato oggi"
    
    return header + "\n".join(parts)

def send_telegram(token: str, chat_id: str, message: str):
    """Invia a Telegram"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        resp = requests.post(url, json=payload, timeout=10)
        print(f"📤 Telegram: {resp.status_code}")
    except Exception as e:
        print(f"❌ Errore Telegram: {e}")

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    print(f"📊 ANALISI GIORNALIERA - {datetime.now().strftime('%d/%m %H:%M')}")
    
    # Carica titoli da CSV
    portfolio, watchlist, descriptions = load_titoli_csv()
    
    print(f"💰 Portafoglio: {len(portfolio)} titoli")
    print(f"👁️  Watchlist: {len(watchlist)} titoli")
    
    # Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    # Analisi portfolio
    portfolio_signals = {}
    for ticker in portfolio:
        signals = analyze_daily_ticker(ticker)
        if signals:
            portfolio_signals[ticker] = signals
    
    # Analisi watchlist
    watchlist_signals = {}
    for ticker in watchlist:
        signals = analyze_daily_ticker(ticker)
        if signals:
            watchlist_signals[ticker] = signals
    
    # Prepara messaggio
    if portfolio_signals or watchlist_signals:
        message = format_daily_message(portfolio_signals, watchlist_signals, descriptions)
        
        if token and chat_id:
            send_telegram(token, chat_id, message)
        else:
            print("ℹ️  Credenziali Telegram mancanti")
            print(message)
    else:
        print("📭 Nessun segnale oggi")
    
    print("✅ Analisi giornaliera completata")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Errore: {e}")
        sys.exit(1)

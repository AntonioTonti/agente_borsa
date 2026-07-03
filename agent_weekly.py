#!/usr/bin/env python3
"""
Agente di Trading - Analisi Settimanale (STESSA LOGICA DEL GIORNALIERO)
Invio: Venerdì 18:00 UTC (19:00 IT)
FEATURES:
- STESSA ANALISI del giornaliero (Heikin Ashi, EMA, RSI, Volume)
- MA su Dati SETTIMANALI invece che giornalieri
- Pallino riassuntivo 🟢/⚪/🔴 vicino al ticker
- Analisi separata per Portafoglio e Watchlist
- Due invii Telegram distinti
"""

import os
import sys
import time
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Configurazione
sys.path.append('.')
from config import (
    load_titoli_csv, DAILY_PERIOD, DAILY_INTERVAL, DAILY_MIN_POINTS
)

# Costanti settimanali
WEEKLY_PERIOD = "1y"      # 1 anno di dati
WEEKLY_INTERVAL = "1wk"   # Barre settimanali
WEEKLY_MIN_POINTS = 20    # Minimo 20 settimane per l'analisi

# ============================================================================
# FUNZIONE HEIKIN ASHI (IDENTICA al giornaliero)
# ============================================================================

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le barre Heikin Ashi a partire da un DataFrame OHLCV
    Restituisce un DataFrame con le colonne HA_Open, HA_High, HA_Low, HA_Close
    """
    ha = pd.DataFrame(index=df.index)
    
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'] + df['Close']) / 2
    
    ha['HA_Close'] = ha_close
    ha['HA_Open'] = ha_open.copy()
    
    for i in range(1, len(df)):
        ha.loc[df.index[i], 'HA_Open'] = (ha.loc[df.index[i-1], 'HA_Open'] + ha.loc[df.index[i-1], 'HA_Close']) / 2
    
    ha['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    for i in range(len(df)):
        ha.loc[df.index[i], 'HA_High'] = max(
            df.loc[df.index[i], 'High'],
            ha.loc[df.index[i], 'HA_Open'],
            ha.loc[df.index[i], 'HA_Close']
        )
        ha.loc[df.index[i], 'HA_Low'] = min(
            df.loc[df.index[i], 'Low'],
            ha.loc[df.index[i], 'HA_Open'],
            ha.loc[df.index[i], 'HA_Close']
        )
    
    return ha

# ============================================================================
# FUNZIONE PALLINO RIASSUNTIVO (IDENTICA al giornaliero)
# ============================================================================

def get_bullet(score: float) -> str:
    """
    Restituisce il pallino in base allo score finale
    🟢 Verde: score > 0.60 (trend rialzista)
    ⚪ Bianco: 0.40 <= score <= 0.60 (laterale/neutro)
    🔴 Rosso: score < 0.40 (trend ribassista)
    """
    if score > 0.60:
        return "🟢"
    elif score < 0.40:
        return "🔴"
    else:
        return "⚪"

# ============================================================================
# INDICATORI SETTIMANALI (IDENTICI al giornaliero)
# ============================================================================

def analyze_weekly_ticker(ticker: str) -> Tuple[List[str], float]:
    """
    Analisi settimanale - STESSA LOGICA del giornaliero
    Restituisce: (segnali, score)
    Score: 0.0 (molto negativo) a 1.0 (molto positivo)
    """
    signals = []
    score = 0.5
    ha_color_score = 0.0
    
    try:
        df = yf.download(ticker, period=WEEKLY_PERIOD, interval=WEEKLY_INTERVAL, progress=False)
        
        if df.empty or len(df) < WEEKLY_MIN_POINTS:
            return signals, score
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        volume = df['Volume']
        
        # ================================================================
        # 1. HEIKIN ASHI (PESO 0.35)
        # ================================================================
        ha = calculate_heikin_ashi(df)
        
        if len(ha) >= 2:
            last_ha_close = float(ha['HA_Close'].iloc[-1])
            prev_ha_close = float(ha['HA_Close'].iloc[-2])
            last_ha_open = float(ha['HA_Open'].iloc[-1])
            
            if last_ha_close > last_ha_open:
                signals.append("🟢 HEIKIN ASHI: BARRA VERDE (Trend rialzista)")
                ha_color_score = 0.35
                
                if last_ha_close > prev_ha_close:
                    signals.append("   ↑ Rafforzamento: Chiusura > Chiusura precedente")
                    ha_color_score += 0.10
            else:
                signals.append("🔴 HEIKIN ASHI: BARRA ROSSA (Trend ribassista)")
                ha_color_score = -0.35
                
                if last_ha_close < prev_ha_close:
                    signals.append("   ↓ Indebolimento: Chiusura < Chiusura precedente")
                    ha_color_score -= 0.10
        
        # ================================================================
        # 2. EMA10 vs MA31 (PESO 0.30)
        # ================================================================
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
                    score += 0.25
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append("📉 MA31 > EMA10 (CROSSOVER DOWN)")
                    score -= 0.25
                elif ema_now > ma_now:
                    signals.append("🟢 EMA10 sopra MA31")
                    score += 0.15
                else:
                    signals.append("🔴 MA31 sopra EMA10")
                    score -= 0.15
        
        # ================================================================
        # 3. RSI (PESO 0.20)
        # ================================================================
        if len(close) >= 15:
            import ta
            rsi = ta.momentum.rsi(close, window=14)
            if len(rsi) > 0:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    signals.append("⚠️ RSI > 70 (IPERCOMPRATO)")
                    score -= 0.15
                elif rsi_val < 30:
                    signals.append("⚠️ RSI < 30 (IPERVENDUTO)")
                    score += 0.10
                elif rsi_val > 60:
                    score += 0.05
                elif rsi_val < 40:
                    score -= 0.05
        
        # ================================================================
        # 4. Volume (PESO 0.15)
        # ================================================================
        if len(volume) >= 10:
            avg_volume = float(volume.tail(10).mean())
            current_volume = float(volume.iloc[-1])
            if current_volume > avg_volume * 1.5:
                signals.append("📊 Volume +50%")
                score += 0.10
            elif current_volume < avg_volume * 0.5:
                score -= 0.05
        
        # ================================================================
        # COMBINAZIONE FINALE SCORE
        # ================================================================
        other_indicators_score = score
        ha_normalized = (ha_color_score + 0.45) / 0.9
        final_score = (ha_normalized * 0.35) + (other_indicators_score * 0.65)
        final_score = max(0.0, min(1.0, final_score))
        
        return signals, round(final_score, 3)
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return signals, 0.5

# ============================================================================
# FUNZIONI DI FORMATTAZIONE REPORT CON PALLINO (IDENTICHE al giornaliero)
# ============================================================================

def create_portfolio_report(results: List[Tuple[str, List[str], float]], descriptions: Dict) -> str:
    """Crea report per portafoglio con pallino"""
    if not results:
        return "💰 *PORTAFOGLIO settimanale* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2])
    
    lines = []
    lines.append("💰 *PORTAFOGLIO settimanale*")
    
    for ticker, signals, score in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        
        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
    
    return "\n".join(lines)

def create_watchlist_report(results: List[Tuple[str, List[str], float]], descriptions: Dict) -> str:
    """Crea report per watchlist con pallino"""
    if not results:
        return "👁️ *WATCHLIST settimanale* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2])
    
    lines = []
    lines.append("👁️ *WATCHLIST settimanale*")
    
    for ticker, signals, score in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        
        lines.append(f"\n{bullet} *{ticker}* - {desc} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
    
    return "\n".join(lines)

# ============================================================================
# FUNZIONI DI INVIO TELEGRAM (IDENTICHE al giornaliero)
# ============================================================================

def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:
    """Invia un messaggio a Telegram con gestione errori"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        MAX_LENGTH = 4096
        
        if len(message) > MAX_LENGTH:
            parts = []
            lines = message.split('\n')
            current_part = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 > MAX_LENGTH:
                    parts.append('\n'.join(current_part))
                    current_part = [line]
                    current_length = len(line)
                else:
                    current_part.append(line)
                    current_length += len(line) + 1
            
            if current_part:
                parts.append('\n'.join(current_part))
        else:
            parts = [message]
        
        for i, part in enumerate(parts):
            payload = {
                "chat_id": chat_id,
                "text": part,
                "parse_mode": "Markdown" if (use_markdown and i == 0) else None,
                "disable_web_page_preview": True,
                "disable_notification": (i > 0)
            }
            
            resp = requests.post(url, json=payload, timeout=15)
            
            if resp.status_code != 200:
                print(f"    ❌ Errore invio parte {i+1}: {resp.status_code}")
                return False
            
            if i < len(parts) - 1:
                time.sleep(0.5)
        
        return True
        
    except Exception as e:
        print(f"❌ Errore invio Telegram: {e}")
        return False

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """Funzione principale - Analisi settimanale con logica giornaliera"""
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("📊 AGENTE DI TRADING - ANALISI SETTIMANALE (Stessa logica del Giornaliero + Pallino)")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        # 1. Caricamento titoli
        print("\n📁 CARICAMENTO TITOLI")
        print("-" * 40)
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        print(f"✅ Titoli caricati:")
        print(f"   • Portafoglio: {len(portfolio)} titoli")
        print(f"   • Watchlist: {len(watchlist)} titoli")
        
        if not portfolio and not watchlist:
            print("❌ Nessun titolo da analizzare")
            return
        
        # 2. Analisi Portafoglio
        portfolio_results = []
        if portfolio:
            print(f"\n💰 ANALISI PORTAFOGLIO (dati SETTIMANALI)")
            print("-" * 40)
            
            for i, ticker in enumerate(portfolio, 1):
                print(f"[{i}/{len(portfolio)}] {ticker}...", end="", flush=True)
                signals, score = analyze_weekly_ticker(ticker)
                
                if signals:
                    portfolio_results.append((ticker, signals, score))
                    print(f" ✅ {len(signals)} segnali (score: {score})")
                else:
                    portfolio_results.append((ticker, [], score))
                    print(f" 📭 nessun segnale (score: {score})")
        
        # 3. Analisi Watchlist
        watchlist_results = []
        if watchlist:
            print(f"\n👁️ ANALISI WATCHLIST (dati SETTIMANALI)")
            print("-" * 40)
            
            for i, ticker in enumerate(watchlist, 1):
                print(f"[{i}/{len(watchlist)}] {ticker}...", end="", flush=True)
                signals, score = analyze_weekly_ticker(ticker)
                
                if signals:
                    watchlist_results.append((ticker, signals, score))
                    print(f" ✅ {len(signals)} segnali (score: {score})")
                else:
                    watchlist_results.append((ticker, [], score))
                    print(f" 📭 nessun segnale (score: {score})")
        
        # 4. Verifica risultati
        print("\n📊 RIEPILOGO RISULTATI")
        print("-" * 40)
        
        portfolio_with_signals = sum(1 for _, signals, _ in portfolio_results if signals)
        watchlist_with_signals = sum(1 for _, signals, _ in watchlist_results if signals)
        
        print(f"Portafoglio con segnali: {portfolio_with_signals}/{len(portfolio)}")
        print(f"Watchlist con segnali: {watchlist_with_sign

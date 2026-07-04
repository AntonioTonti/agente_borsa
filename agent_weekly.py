#!/usr/bin/env python3
"""
Agente di Trading - Analisi Settimanale (STESSA LOGICA DEL GIORNALIERO)
Invio: Venerdì 18:00 UTC (19:00 IT)
FEATURES:
- STESSA ANALISI del giornaliero (Heikin Ashi, EMA, RSI, Volume)
- MA su Dati SETTIMANALI invece che giornalieri
- Stima Trend (3 settimane) con Target e Stop Loss
- Pallino riassuntivo 🟢/⚪/🔴 a destra del nome
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
from analysis_utils import (
    calculate_heikin_ashi,
    get_bullet,
    calculate_trend_estimate,
    format_trend_line
)

# Costanti settimanali
WEEKLY_PERIOD = "1y"      # 1 anno di dati
WEEKLY_INTERVAL = "1wk"   # Barre settimanali
WEEKLY_MIN_POINTS = 20    # Minimo 20 settimane per l'analisi

# ============================================================================
# INDICATORI SETTIMANALI (IDENTICI al giornaliero)
# ============================================================================

def analyze_weekly_ticker(ticker: str) -> Tuple[List[str], float, Dict]:
    """
    Analisi settimanale - STESSA LOGICA del giornaliero
    Restituisce: (segnali, score, dati_aggiuntivi)
    """
    signals = []
    score = 0.5
    ha_color_score = 0.0
    extra_data = {}
    
    try:
        df = yf.download(ticker, period=WEEKLY_PERIOD, interval=WEEKLY_INTERVAL, progress=False)
        
        if df.empty or len(df) < WEEKLY_MIN_POINTS:
            return signals, score, extra_data
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        volume = df['Volume']
        
        # ================================================================
        # STIMA TREND (3 settimane)
        # ================================================================
        if len(close) >= 6:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=3)
            extra_data = {
                'var_percent': var_percent,
                'target_price': target_price,
                'stop_loss': stop_loss
            }
        
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
        
        return signals, round(final_score, 3), extra_data
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return signals, 0.5, extra_data

# ============================================================================
# FUNZIONI DI FORMATTAZIONE REPORT CON PALLINO A DESTRA E TREND
# ============================================================================

def create_portfolio_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    """Crea report per portafoglio con pallino a destra e stima trend"""
    if not results:
        return "💰 *PORTAFOGLIO SETTIMANALE* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2])
    
    lines = []
    lines.append("💰 *PORTAFOGLIO SETTIMANALE*")
    
    for ticker, signals, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        
        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
        
        # Aggiungi stima trend se disponibile
        if extra_data and 'var_percent' in extra_data:
            trend_line = format_trend_line(
                extra_data['var_percent'],
                extra_data['target_price'],
                extra_data['stop_loss']
            )
            lines.append(trend_line)
        
        lines.append("----------------------------")
    
    return "\n".join(lines)

def create_watchlist_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    """Crea report per watchlist con pallino a destra e stima trend"""
    if not results:
        return "👁️ *OSSERVATI SETTIMANALI* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2])
    
    lines = []
    lines.append("👁️ *OSSERVATI SETTIMANALI*")

#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera
PESI E SOGLIE AGGIORNATI (TOTALE 100%):
- Stima Trend 7gg (30%) [Soglia: ±3.0%]
- EMA10 vs MA31 (30%)
- Heikin Ashi (20%)
- Volume (5%)
- Chiusura vs Chiusura Prec. (5%)
- ZigZag (5%)
- RSI 14 (5%)
"""

import os
import sys
import time
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta
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

def calculate_zigzag_trend(df: pd.DataFrame, deviation_pct: float = 5.0) -> int:
    """
    Calcola l'ultimo trend dell'indicatore ZigZag.
    Ritorna: 1 se Rialzista, -1 se Ribassista, 0 se insufficiente.
    """
    if len(df) < 20:
        return 0
        
    highs = df['High'].squeeze().values
    lows = df['Low'].squeeze().values
    
    last_pivot_val = highs[0]
    last_pivot_type = 'H'
    
    trends = []
    thresh = deviation_pct / 100.0
    
    for i in range(1, len(df)):
        if last_pivot_type == 'H':
            if highs[i] > last_pivot_val:
                last_pivot_val = highs[i]
            elif lows[i] <= last_pivot_val * (1.0 - thresh):
                last_pivot_val = lows[i]
                last_pivot_type = 'L'
                trends.append(-1)
        else:
            if lows[i] < last_pivot_val:
                last_pivot_val = lows[i]
            elif highs[i] >= last_pivot_val * (1.0 + thresh):
                last_pivot_val = highs[i]
                last_pivot_type = 'H'
                trends.append(1)
                
    if not trends:
        return 1 if last_pivot_type == 'H' else -1
    return trends[-1]

def analyze_daily_ticker(ticker: str) -> Tuple[List[str], float, Dict]:
    """
    Analisi giornaliera con esposizione completa di tutti i parametri ed errori numerici
    """
    signals = []
    
    # Default neutrali
    trend_score = 0.5
    ema_ma_score = 0.5
    ha_score = 0.5
    vol_score = 0.5
    close_change_score = 0.5
    zigzag_score = 0.5
    rsi_score = 0.5
    
    extra_data = {}
    
    try:
        # Download dati con storico di 6 mesi
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            return signals, 0.5, extra_data
        
        # Gestione MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        
        # 1. STIMA TREND 7 GIORNI (PESO 30%)
        if len(close) >= 10:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
            extra_data = {
                'var_percent': var_percent,
                'target_price': target_price,
                'stop_loss': stop_loss
            }
            if var_percent > 3.0:
                trend_score = 1.0
            elif var_percent > 0.0:
                trend_score = 0.75
            elif var_percent == 0.0:
                trend_score = 0.50
            elif var_percent > -3.0:
                trend_score = 0.25
            else:
                trend_score = 0.0

        # 2. EMA10 vs MA31 (PESO 30%) - Mostra SEMPRE i valori calcolati
        if len(close) >= 32:
            ema10 = ta.trend.ema_indicator(close, window=10)
            ma31 = ta.trend.sma_indicator(close, window=31)
            
            clean_ema = ema10.dropna()
            clean_ma = ma31.dropna()
            
            if len(clean_ema) > 1 and len(clean_ma) > 1:
                ema_now = float(clean_ema.iloc[-1])
                ma_now = float(clean_ma.iloc[-1])
                ema_prev = float(clean_ema.iloc[-2])
                ma_prev = float(clean_ma.iloc[-2])
                
                # Format dinamico decimali (per penny stock come Tessellis)
                fmt = ".4f" if ema_now < 1.0 else ".2f"
                
                if ema_now > ma_now and ema_prev <= ma_prev:
                    signals.append(f"📈 EMA10 ({ema_now:{fmt}}) > MA31 ({ma_now:{fmt}}) [CROSSOVER UP]")
                    ema_ma_score = 1.0
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append(f"📉 MA31 ({ma_now:{fmt}}) > EMA10 ({ema_now:{fmt}}) [CROSSOVER DOWN]")
                    ema_ma_score = 0.0
                elif ema_now > ma_now:
                    signals.append(f"🟢 EMA10 ({ema_now:{fmt}}) sopra MA31 ({ma_now:{fmt}})")
                    ema_ma_score = 0.75
                else:
                    signals.append(f"🔴 MA31 ({ma_now:{fmt}}) sopra EMA10 ({ema_now:{fmt}})")
                    ema_ma_score = 0.25

        # 3. HEIKIN ASHI (PESO 20%) - Mostra SEMPRE lo stato
        ha = calculate_heikin_ashi(df)
        if len(ha) >= 2:
            last_ha_close = float(ha['HA_Close'].iloc[-1])
            prev_ha_close = float(ha['HA_Close'].iloc[-2])
            last_ha_open = float(ha['HA_Open'].iloc[-1])
            
            if last_ha_close > last_ha_open:
                msg = "🟢 HEIKIN ASHI: BARRA VERDE"
                if last_ha_close > prev_ha_close:
                    msg += " (↑ Rafforzamento)"
                    ha_score = 1.0
                else:
                    ha_score = 0.85
                signals.append(msg)
            else:
                msg = "🔴 HEIKIN ASHI: BARRA ROSSA"
                if last_ha_close < prev_ha_close:
                    msg += " (↓ Indebolimento)"
                    ha_score = 0.0
                else:
                    ha_score = 0.15
                signals.append(msg)

        # 4. RSI 14 (PESO 5%) - Mostra SEMPRE il valore dell'RSI
        if len(close) >= 15:
            rsi = ta.momentum.rsi(close, window=14).dropna()
            if not rsi.empty:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    signals.append(f"⚠️ RSI: {rsi_val:.1f} (IPERCOMPRATO)")
                    rsi_score = 0.15
                elif rsi_val < 30:
                    signals.append(f"⚠️ RSI: {rsi_val:.1f} (IPERVENDUTO)")
                    rsi_score = 0.85
                else:
                    signals.append(f"📊 RSI: {rsi_val:.1f} (Neutro)")
                    if rsi_val > 60:
                        rsi_score = 0.65
                    elif rsi_val < 40:
                        rsi_score = 0.35
                    else:
                        rsi_score = 0.50

        # 5. VOLUME (PESO 5%) - Mostra SEMPRE il confronto con la media
        if len(volume) >= 10:
            avg_vol = float(volume.tail(10).mean())
            curr_vol = float(volume.iloc[-1])
            diff_pct = ((curr_vol - avg_vol) / avg_vol * 100.0) if avg_vol > 0 else 0.0
            
            if curr_vol > avg_vol * 1.5:
                signals.append(f"📊 Volume: +{diff_pct:.0f}% vs media 10gg")
                vol_score = 0.80
            elif curr_vol < avg_vol * 0.5:
                signals.append(f"📊 Volume: {diff_pct:.0f}% vs media 10gg")
                vol_score = 0.30
            else:
                signals.append(f"📊 Volume: nella media ({diff_pct:+.0f}%)")
                vol_score = 0.50

        # 6. ZIGZAG (PESO 5%) - Mostra SEMPRE lo stato
        zz_trend = calculate_zigzag_trend(df, deviation_pct=5.0)
        if zz_trend == 1:
            signals.append("⚡ ZIGZAG: Rialzista")
            zigzag_score = 1.0
        elif zz_trend == -1:
            signals.append("⚡ ZIGZAG: Ribassista")
            zigzag_score = 0.0
        else:
            signals.append("⚡ ZIGZAG: Incertezza")
            zigzag_score = 0.5

        # 7. CHIUSURA VS PRECEDENTE (PESO 5%)
        if len(close) >= 2:
            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            pct_change = ((last_close - prev_close) / prev_close) * 100.0
            
            if pct_change > 0.5:
                close_change_score = 1.0
            elif pct_change > 0:
                close_change_score = 0.75
            elif pct_change == 0:
                close_change_score = 0.50
            elif pct_change > -0.5:
                close_change_score = 0.25
            else:
                close_change_score = 0.0

        # COMBINAZIONE FINALE SCORE
        final_score = (
            (trend_score * 0.30) +
            (ema_ma_score * 0.30) +
            (ha_score * 0.20) +
            (vol_score * 0.05) +
            (close_change_score * 0.05) +
            (zigzag_score * 0.05) +
            (rsi_score * 0.05)
        )
        final_score = max(0.0, min(1.0, final_score))
        
        return signals, round(final_score, 3), extra_data
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return signals, 0.5, extra_data

def create_portfolio_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    if not results:
        return "💰 *PORTAFOGLIO GIORNALIERO* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = ["💰 *PORTAFOGLIO GIORNALIERO*"]
    
    for ticker, signals, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append("  📭 Nessun segnale rilevato")
            
        if extra_data and 'var_percent' in extra_data:
            lines.append(format_trend_line(extra_data['var_percent'], extra_data['target_price'], extra_data['stop_loss']))
        lines.append("----------------------------")
        
    return "\n".join(lines)

def create_watchlist_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    if not results:
        return "👁️ *OSSERVATI GIORNALIERI* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = ["👁️ *OSSERVATI GIORNALIERI*"]
    
    for ticker, signals, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append("  📭 Nessun segnale rilevato")
            
        if extra_data and 'var_percent' in extra_data:
            lines.append(format_trend_line(extra_data['var_percent'], extra_data['target_price'], extra_data['stop_loss']))
        lines.append("----------------------------")
        
    return "\n".join(lines)

def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown" if use_markdown else None,
            "disable_web_page_preview": True
        }
        resp = requests.post(url, json=payload, timeout=15)
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Errore invio Telegram: {e}")
        return False

def main():
    start_time = time.time()
    try:
        print("=" * 60)
        print("📊 AGENTE DI TRADING - ANALISI GIORNALIERA COMPLETA")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        
        portfolio_results = []
        if portfolio:
            print("\n💰 ANALISI PORTAFOGLIO")
            for ticker in portfolio:
                signals, score, extra_data = analyze_daily_ticker(ticker)
                portfolio_results.append((ticker, signals, score, extra_data))
                
        watchlist_results = []
        if watchlist:
            print("\n👁️ ANALISI WATCHLIST")
            for ticker in watchlist:
                signals, score, extra_data = analyze_daily_ticker(ticker)
                watchlist_results.append((ticker, signals, score, extra_data))
                
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            if portfolio_results:
                send_telegram_message(token, chat_id, create_portfolio_daily_report(portfolio_results, descriptions))
            if watchlist_results:
                send_telegram_message(token, chat_id, create_watchlist_daily_report(watchlist_results, descriptions))
                
        print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")

if __name__ == "__main__":
    main()

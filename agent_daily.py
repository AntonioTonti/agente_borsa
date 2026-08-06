#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera con Heikin Ashi, Indicatori e ZigZag
Invio: 12:00 e 17:00 UTC (13:00 e 18:00 IT)

PESI SCORING AGGIORNATI (TOTALE 100%):
- Heikin Ashi (25%)
- EMA10 vs MA31 (25%)
- Stima Trend 7gg (20%)
- Volume (10%)
- Variazione Chiusura Giornaliera (10%)
- ZigZag (5%)
- RSI (5%)
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

# ============================================================================
# FUNZIONE DI SUPPORTO PER CALCOLO ZIGZAG
# ============================================================================
def calculate_zigzag_trend(df: pd.DataFrame, deviation_pct: float = 5.0) -> int:
    """
    Calcola l'ultimo trend dell'indicatore ZigZag basato su massimi e minimi.
    Ritorna: 1 se l'ultimo segmento è Rialzista, -1 se Ribassista, 0 se insufficiente.
    """
    if len(df) < 20:
        return 0
        
    highs = df['High'].values
    lows = df['Low'].values
    
    last_pivot_val = highs[0]
    last_pivot_type = 'H'  # H = High, L = Low
    
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

# ============================================================================
# INDICATORI GIORNALIERI AGGIORNATI
# ============================================================================

def analyze_daily_ticker(ticker: str) -> Tuple[List[str], float, Dict]:
    """
    Analisi rapida giornaliera con calcolo score pesato e stima trend
    Restituisce: (segnali, score, dati_aggiuntivi)
    """
    signals = []
    
    # Inizializzazione punteggi parziali (default neutrale 0.5)
    ha_score = 0.5
    ema_ma_score = 0.5
    trend_score = 0.5
    vol_score = 0.5
    close_change_score = 0.5
    zigzag_score = 0.5
    rsi_score = 0.5
    
    extra_data = {}
    
    try:
        df = yf.download(ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL, progress=False)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            return signals, 0.5, extra_data
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        volume = df['Volume']
        
        # ================================================================
        # 1. STIMA TREND 7 GIORNI (PESO 0.20)
        # ================================================================
        if len(close) >= 10:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
            extra_data = {
                'var_percent': var_percent,
                'target_price': target_price,
                'stop_loss': stop_loss
            }
            if var_percent > 1.5:
                trend_score = 1.0
            elif var_percent > 0.0:
                trend_score = 0.75
            elif var_percent == 0.0:
                trend_score = 0.50
            elif var_percent > -1.5:
                trend_score = 0.25
            else:
                trend_score = 0.0
        
        # ================================================================
        # 2. HEIKIN ASHI (PESO 0.25)
        # ================================================================
        ha = calculate_heikin_ashi(df)
        if len(ha) >= 2:
            last_ha_close = float(ha['HA_Close'].iloc[-1])
            prev_ha_close = float(ha['HA_Close'].iloc[-2])
            last_ha_open = float(ha['HA_Open'].iloc[-1])
            
            if last_ha_close > last_ha_open:
                signals.append("🟢 HEIKIN ASHI: BARRA VERDE (Trend rialzista)")
                ha_score = 0.85
                if last_ha_close > prev_ha_close:
                    signals.append("   ↑ Rafforzamento: Chiusura > Chiusura precedente")
                    ha_score = 1.0
            else:
                signals.append("🔴 HEIKIN ASHI: BARRA ROSSA (Trend ribassista)")
                ha_score = 0.15
                if last_ha_close < prev_ha_close:
                    signals.append("   ↓ Indebolimento: Chiusura < Chiusura precedente")
                    ha_score = 0.0
        
        # ================================================================
        # 3. EMA10 vs MA31 (PESO 0.25)
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
                    ema_ma_score = 1.0
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append("📉 MA31 > EMA10 (CROSSOVER DOWN)")
                    ema_ma_score = 0.0
                elif ema_now > ma_now:
                    signals.append("🟢 EMA10 sopra MA31")
                    ema_ma_score = 0.75
                else:
                    signals.append("🔴 MA31 sopra EMA10")
                    ema_ma_score = 0.25

        # ================================================================
        # 4. Volume (PESO 0.10)
        # ================================================================
        if len(volume) >= 10:
            avg_volume = float(volume.tail(10).mean())
            current_volume = float(volume.iloc[-1])
            if current_volume > avg_volume * 1.5:
                signals.append("📊 Volume +50%")
                vol_score = 0.80
            elif current_volume < avg_volume * 0.5:
                vol_score = 0.30

        # ================================================================
        # 5. CHIUSURA VS CHIUSURA PRECEDENTE (PESO 0.10)
        # ================================================================
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

        # ================================================================
        # 6. ZIGZAG (PESO 0.05)
        # ================================================================
        zz_trend = calculate_zigzag_trend(df, deviation_pct=5.0)
        if zz_trend == 1:
            signals.append("⚡ ZIGZAG: Segmento Rialzista Attivo")
            zigzag_score = 1.0
        elif zz_trend == -1:
            signals.append("⚡ ZIGZAG: Segmento Ribassista Attivo")
            zigzag_score = 0.0

        # ================================================================
        # 7. RSI (PESO 0.05)
        # ================================================================
        if len(close) >= 15:
            import ta
            rsi = ta.momentum.rsi(close, window=14)
            if len(rsi) > 0:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    signals.append("⚠️ RSI > 70 (IPERCOMPRATO)")
                    rsi_score = 0.15
                elif rsi_val < 30:
                    signals.append("⚠️ RSI < 30 (IPERVENDUTO)")
                    rsi_score = 0.85
                elif rsi_val > 60:
                    rsi_score = 0.65
                elif rsi_val < 40:
                    rsi_score = 0.35

        # ================================================================
        # COMBINAZIONE FINALE SCORE (PESI AGGIORNATI 100%)
        # ================================================================
        final_score = (
            (ha_score * 0.25) +
            (ema_ma_score * 0.25) +
            (trend_score * 0.20) +
            (vol_score * 0.10) +
            (close_change_score * 0.10) +
            (zigzag_score * 0.05) +
            (rsi_score * 0.05)
        )
        final_score = max(0.0, min(1.0, final_score))
        
        return signals, round(final_score, 3), extra_data
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return signals, 0.5, extra_data

# ============================================================================
# FUNZIONI DI FORMATTAZIONE REPORT
# ============================================================================

def create_portfolio_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    if not results:
        return "💰 *PORTAFOGLIO GIORNALIERO* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2])
    
    lines = []
    lines.append("💰 *PORTAFOGLIO GIORNALIERO*")
    
    for ticker, signals, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        
        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
        
        if extra_data and 'var_percent' in extra_data:
            trend_line = format_trend_line(
                extra_data['var_percent'],
                extra_data['target_price'],
                extra_data['stop_loss']
            )
            lines.append(trend_line)
        
        lines.append("----------------------------")
    
    return "\n".join(lines)

def create_watchlist_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    if not results:
        return "👁️ *OSSERVATI GIORNALIERI* - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2])
    
    lines = []
    lines.append("👁️ *OSSERVATI GIORNALIERI*")
    
    for ticker, signals, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        
        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
        
        if extra_data and 'var_percent' in extra_data:
            trend_line = format_trend_line(
                extra_data['var_percent'],
                extra_data['target_price'],
                extra_data['stop_loss']
            )
            lines.append(trend_line)
        
        lines.append("----------------------------")
    
    return "\n".join(lines)

# ============================================================================
# FUNZIONI DI INVIO TELEGRAM
# ============================================================================

def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:
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
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("📊 AGENTE DI TRADING - ANALISI GIORNALIERA CON NUOVI PESI")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        print("\n📁 CARICAMENTO TITOLI")
        print("-" * 40)
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        print(f"✅ Titoli caricati:")
        print(f"    • Portafoglio: {len(portfolio)} titoli")
        print(f"    • Watchlist: {len(watchlist)} titoli")
        
        if not portfolio and not watchlist:
            print("❌ Nessun titolo da analizzare")
            return
        
        portfolio_results = []
        if portfolio:
            print(f"\n💰 ANALISI PORTAFOGLIO")
            print("-" * 40)
            
            for i, ticker in enumerate(portfolio, 1):
                print(f"[{i}/{len(portfolio)}] {ticker}...", end="", flush=True)
                signals, score, extra_data = analyze_daily_ticker(ticker)
                
                if signals:
                    portfolio_results.append((ticker, signals, score, extra_data))
                    print(f" ✅ {len(signals)} segnali (score: {score})")
                else:
                    portfolio_results.append((ticker, [], score, extra_data))
                    print(f" 📭 nessun segnale (score: {score})")
        
        watchlist_results = []
        if watchlist:
            print(f"\n👁️ ANALISI WATCHLIST")
            print("-" * 40)
            
            for i, ticker in enumerate(watchlist, 1):
                print(f"[{i}/{len(watchlist)}] {ticker}...", end="", flush=True)
                signals, score, extra_data = analyze_daily_ticker(ticker)
                
                if signals:
                    watchlist_results.append((ticker, signals, score, extra_data))
                    print(f" ✅ {len(signals)} segnali (score: {score})")
                else:
                    watchlist_results.append((ticker, [], score, extra_data))
                    print(f" 📭 nessun segnale (score: {score})")
        
        print("\n📊 RIEPILOGO RISULTATI")
        print("-" * 40)
        
        portfolio_with_signals = sum(1 for _, signals, _, _ in portfolio_results if signals)
        watchlist_with_signals = sum(1 for _, signals, _, _ in watchlist_results if signals)
        
        print(f"Portafoglio con segnali: {portfolio_with_signals}/{len(portfolio)}")
        print(f"Watchlist con segnali: {watchlist_with_signals}/{len(watchlist)}")
        
        if not portfolio_with_signals and not watchlist_with_signals:
            print("📭 Nessun segnale da inviare oggi")
            return
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            print("⚠️ Credenziali Telegram non configurate")
            return
        
        print("\n📤 INVIO REPORT TELEGRAM")
        print("-" * 40)
        
        if portfolio_results:
            print("\n1️⃣ INVIO PORTAFOGLIO")
            portfolio_message = create_portfolio_daily_report(portfolio_results, descriptions)
            success = send_telegram_message(token, chat_id, portfolio_message, use_markdown=True)
            if success:
                print("✅ Portafoglio inviato con successo!")
            time.sleep(2)
        
        if watchlist_results:
            print("\n2️⃣ INVIO WATCHLIST")
            watchlist_message = create_watchlist_daily_report(watchlist_results, descriptions)
            success = send_telegram_message(token, chat_id, watchlist_message, use_markdown=True)
            if success:
                print("✅ Watchlist inviata con successo!")
        
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🏁 ANALISI GIORNALIERA COMPLETATA")
        print(f"Tempo impiegato: {elapsed_time:.1f} secondi")
        print("=" * 60)
        
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

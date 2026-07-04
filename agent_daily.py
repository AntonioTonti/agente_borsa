#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera con Heikin Ashi e Pallino
Invio: 12:00 e 17:00 UTC (13:00 e 18:00 IT)
FEATURES:
- Heikin Ashi (peso 0.35) - barra verde/rossa
- EMA10 vs MA31 (peso 0.30)
- RSI (peso 0.20)
- Volume (peso 0.15)
- Pallino riassuntivo 🟢/⚪/🔴 a destra del nome
- Stima Trend (7 giorni) con Target e Stop Loss
- Analisi separata per Portafoglio e Watchlist
- Due invii Telegram distinti
- Titoli ordinati dal peggiore al migliore
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
# INDICATORI GIORNALIERI (CON HEIKIN ASHI)
# ============================================================================

def analyze_daily_ticker(ticker: str) -> Tuple[List[str], float, Dict]:
    """
    Analisi rapida giornaliera con calcolo score e stima trend
    Restituisce: (segnali, score, dati_aggiuntivi)
    """
    signals = []
    score = 0.5
    ha_color_score = 0.0
    extra_data = {}  # Per dati di trend, target, stop loss
    
    try:
        # Download dati
        df = yf.download(ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL, progress=False)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            return signals, score, extra_data
        
        # Pulizia dati
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        volume = df['Volume']
        
        # ================================================================
        # STIMA TREND (7 giorni)
        # ================================================================
        if len(close) >= 10:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
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

def create_portfolio_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    """Crea report giornaliero per portafoglio con pallino a destra e stima trend"""
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

def create_watchlist_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    """Crea report giornaliero per watchlist con pallino a destra e stima trend"""
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

# ============================================================================
# FUNZIONI DI INVIO TELEGRAM
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
    """Funzione principale - Analisi giornaliera"""
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("📊 AGENTE DI TRADING - ANALISI GIORNALIERA (Heikin Ashi + Pallino + Trend)")
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
        
        # 3. Analisi Watchlist
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
        
        # 4. Verifica risultati
        print("\n📊 RIEPILOGO RISULTATI")
        print("-" * 40)
        
        portfolio_with_signals = sum(1 for _, signals, _, _ in portfolio_results if signals)
        watchlist_with_signals = sum(1 for _, signals, _, _ in watchlist_results if signals)
        
        print(f"Portafoglio con segnali: {portfolio_with_signals}/{len(portfolio)}")
        print(f"Watchlist con segnali: {watchlist_with_signals}/{len(watchlist)}")
        
        if not portfolio_with_signals and not watchlist_with_signals:
            print("📭 Nessun segnale da inviare oggi")
            return
        
        # 5. Invio Telegram
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            print("⚠️ Credenziali Telegram non configurate")
            print("   TELEGRAM_BOT_TOKEN:", "✅" if token else "❌")
            print("   TELEGRAM_CHAT_ID:", "✅" if chat_id else "❌")
            return
        
        print("\n📤 INVIO REPORT TELEGRAM")
        print("-" * 40)
        
        # INVIO 1: PORTAFOGLIO
        if portfolio_results:
            print("\n1️⃣ INVIO PORTAFOGLIO")
            portfolio_message = create_portfolio_daily_report(portfolio_results, descriptions)
            print(f"   Lunghezza: {len(portfolio_message)} caratteri")
            
            success = send_telegram_message(token, chat_id, portfolio_message, use_markdown=True)
            
            if success:
                print("✅ Portafoglio inviato con successo!")
            else:
                print("❌ Errore nell'invio portafoglio")
            
            time.sleep(2)
        
        # INVIO 2: WATCHLIST
        if watchlist_results:
            print("\n2️⃣ INVIO WATCHLIST")
            watchlist_message = create_watchlist_daily_report(watchlist_results, descriptions)
            print(f"   Lunghezza: {len(watchlist_message)} caratteri")
            
            success = send_telegram_message(token, chat_id, watchlist_message, use_markdown=True)
            
            if success:
                print("✅ Watchlist inviata con successo!")
            else:
                print("❌ Errore nell'invio watchlist")
        
        # 6. Statistiche finali
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🏁 ANALISI GIORNALIERA COMPLETATA")
        print("-" * 40)
        print(f"Tempo impiegato: {elapsed_time:.1f} secondi")
        print(f"Ora completamento: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ INTERROTTO DALL'UTENTE")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ESECUZIONE
# ============================================================================

if __name__ == "__main__":
    main()

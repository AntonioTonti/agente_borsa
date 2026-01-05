#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera
Invio: 12:00 e 17:00 UTC (13:00 e 18:00 IT)
FEATURES:
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
from typing import List, Dict, Tuple

# Configurazione
sys.path.append('.')
from config import (
    load_titoli_csv, DAILY_PERIOD, DAILY_INTERVAL, DAILY_MIN_POINTS
)

# ============================================================================
# INDICATORI GIORNALIERI (RAPIDI) - MODIFICATO PER ORDINAMENTO
# ============================================================================

def analyze_daily_ticker(ticker: str) -> Tuple[List[str], float]:
    """
    Analisi rapida giornaliera con calcolo score
    Restituisce: (segnali, score)
    Score: 0.0 (molto negativo) a 1.0 (molto positivo)
    """
    signals = []
    score = 0.5  # Score base neutro
    
    try:
        # Download dati
        df = yf.download(ticker, period=DAILY_PERIOD, interval=DAILY_INTERVAL, progress=False)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            return signals, score
        
        # Pulizia dati
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        close = df['Close']
        volume = df['Volume']
        
        # 1. EMA10 vs MA31 (indicatore principale) - PESO 0.5
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
                    score += 0.25  # Forte positivo
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append("📉 MA31 > EMA10 (CROSSOVER DOWN)")
                    score -= 0.25  # Forte negativo
                elif ema_now > ma_now:
                    signals.append("🟢 EMA10 sopra MA31")
                    score += 0.15   # Positivo
                else:
                    signals.append("🔴 MA31 sopra EMA10")
                    score -= 0.15   # Negativo
        
        # 2. RSI giornaliero - PESO 0.3
        if len(close) >= 15:
            import ta
            rsi = ta.momentum.rsi(close, window=14)
            if len(rsi) > 0:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    signals.append("⚠️  RSI > 70 (IPERCOMPRATO)")
                    score -= 0.15   # Negativo (ipercomprato)
                elif rsi_val < 30:
                    signals.append("⚠️  RSI < 30 (IPERVENDUTO)")
                    score += 0.10   # Leggermente positivo (potenziale rimbalzo)
                elif rsi_val > 60:
                    score += 0.05   # Leggermente positivo
                elif rsi_val < 40:
                    score -= 0.05   # Leggermente negativo
        
        # 3. Volume vs media - PESO 0.2
        if len(volume) >= 10:
            avg_volume = float(volume.tail(10).mean())
            current_volume = float(volume.iloc[-1])
            if current_volume > avg_volume * 1.5:
                signals.append("📊 Volume +50%")
                score += 0.10   # Positivo (interesse)
            elif current_volume < avg_volume * 0.5:
                score -= 0.05   # Negativo (scarso interesse)
        
        # Normalizza score tra 0.0 e 1.0
        score = max(0.0, min(1.0, score))
        
        return signals, round(score, 3)
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return signals, 0.5  # Score neutro in caso di errore

# ============================================================================
# FUNZIONI DI FORMATTAZIONE REPORT
# ============================================================================

def create_portfolio_daily_report(results: List[Tuple[str, List[str], float]], descriptions: Dict) -> str:
    """Crea report giornaliero per portafoglio"""
    if not results:
        return "💰 *PORTAFOGLIO* - Nessun segnale oggi"
    
    # Ordina dal PEGGIORE (score basso) al MIGLIORE (score alto)
    sorted_results = sorted(results, key=lambda x: x[2])  # x[2] = score
    
    lines = []
    lines.append("💰 *PORTAFOGLIO*")
    
    for ticker, signals, score in sorted_results:
        desc = descriptions.get(ticker, ticker)
        lines.append(f"\n*{ticker}* - {desc}")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
    
    return "\n".join(lines)

def create_watchlist_daily_report(results: List[Tuple[str, List[str], float]], descriptions: Dict) -> str:
    """Crea report giornaliero per watchlist"""
    if not results:
        return "👁️  *WATCHLIST* - Nessun segnale oggi"
    
    # Ordina dal PEGGIORE (score basso) al MIGLIORE (score alto)
    sorted_results = sorted(results, key=lambda x: x[2])  # x[2] = score
    
    lines = []
    lines.append("👁️  *WATCHLIST*")
    
    for ticker, signals, score in sorted_results:
        desc = descriptions.get(ticker, ticker)
        lines.append(f"\n*{ticker}* - {desc}")
        
        if signals:
            for signal in signals:
                lines.append(f"  {signal}")
        else:
            lines.append(f"  📭 Nessun segnale rilevato")
    
    return "\n".join(lines)

# ============================================================================
# FUNZIONI DI INVIO TELEGRAM
# ============================================================================

def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:
    """Invia un messaggio a Telegram con gestione errori"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Limite Telegram
        MAX_LENGTH = 4096
        
        # Se messaggio troppo lungo, dividi
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
        
        # Invia tutte le parti
        for i, part in enumerate(parts):
            payload = {
                "chat_id": chat_id,
                "text": part,
                "parse_mode": "Markdown" if (use_markdown and i == 0) else None,
                "disable_web_page_preview": True,
                "disable_notification": (i > 0)  # Solo prima parte fa notifica
            }
            
            resp = requests.post(url, json=payload, timeout=15)
            
            if resp.status_code != 200:
                print(f"    ❌ Errore invio parte {i+1}: {resp.status_code}")
                return False
            
            # Pausa tra le parti
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
        print("📊 AGENTE DI TRADING - ANALISI GIORNALIERA")
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
                signals, score = analyze_daily_ticker(ticker)
                
                if signals:
                    portfolio_results.append((ticker, signals, score))
                    print(f" ✅ {len(signals)} segnali (score: {score})")
                else:
                    portfolio_results.append((ticker, [], score))
                    print(f" 📭 nessun segnale (score: {score})")
        
        # 3. Analisi Watchlist
        watchlist_results = []
        if watchlist:
            print(f"\n👁️  ANALISI WATCHLIST")
            print("-" * 40)
            
            for i, ticker in enumerate(watchlist, 1):
                print(f"[{i}/{len(watchlist)}] {ticker}...", end="", flush=True)
                signals, score = analyze_daily_ticker(ticker)
                
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
        print(f"Watchlist con segnali: {watchlist_with_signals}/{len(watchlist)}")
        
        if not portfolio_with_signals and not watchlist_with_signals:
            print("📭 Nessun segnale da inviare oggi")
            return
        
        # 5. Invio Telegram
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            print("⚠️  Credenziali Telegram non configurate")
            print("   TELEGRAM_BOT_TOKEN:", "✅" if token else "❌")
            print("   TELEGRAM_CHAT_ID:", "✅" if chat_id else "❌")
            return
        
        print("\n📤 INVIO REPORT TELEGRAM")
        print("-" * 40)
        
        # INVIO 1: PORTAFOGLIO
        if portfolio_results:
            print("\n1️⃣  INVIO PORTAFOGLIO")
            portfolio_message = create_portfolio_daily_report(portfolio_results, descriptions)
            print(f"   Lunghezza: {len(portfolio_message)} caratteri")
            
            success = send_telegram_message(token, chat_id, portfolio_message, use_markdown=True)
            
            if success:
                print("✅ Portafoglio inviato con successo!")
            else:
                print("❌ Errore nell'invio portafoglio")
            
            # Pausa tra i due invii
            time.sleep(2)
        
        # INVIO 2: WATCHLIST
        if watchlist_results:
            print("\n2️⃣  INVIO WATCHLIST")
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
        print("\n\n⏹️  INTERROTTO DALL'UTENTE")
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

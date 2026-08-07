#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera (Focus Direzionalità e Forza)
PESI E SOGLIE COMPLETI (TOTALE 100%):
1. EMA10 vs MA31 (20%)
2. Stima Trend 7gg (15%) [Soglia: ±3.0%]
3. Delta % EMA10 vs MA31 vs Media 3 Mesi (15%)
4. Heikin Ashi - Forza/Estensione Corpo vs Media 3 Mesi (15%)
5. Heikin Ashi - Stato/Colore & Ombre (10%)
6. Volume vs Media 3 Mesi (5%)
7. Chiusura vs Chiusura Prec. (5%)
8. ZigZag (5%)
9. RSI 14 (5%)
10. MACD (5%)
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

# Configurazione ambiente
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
    Analisi giornaliera a 10 parametri con calibrazione su 3 mesi (63 sessioni)
    e analisi tecnica completa sulle candele Heikin Ashi.
    """
    signals = []
    
    # Default neutrali
    ema_ma_score = 0.5
    trend_score = 0.5
    ema_ma_delta_score = 0.5
    ha_force_score = 0.5
    ha_state_score = 0.5
    vol_score = 0.5
    close_change_score = 0.5
    zigzag_score = 0.5
    rsi_score = 0.5
    macd_score = 0.5
    
    extra_data = {}
    
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            return signals, 0.5, extra_data
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        
        # 1. EMA10 vs MA31 (PESO 20%)
        ema_now = None
        ma_now = None
        ema_series = None
        ma_series = None
        
        if len(close) >= 32:
            ema10 = ta.trend.ema_indicator(close, window=10)
            ma31 = ta.trend.sma_indicator(close, window=31)
            
            clean_ema = ema10.dropna()
            clean_ma = ma31.dropna()
            
            if len(clean_ema) > 1 and len(clean_ma) > 1:
                ema_series = clean_ema
                ma_series = clean_ma
                ema_now = float(clean_ema.iloc[-1])
                ma_now = float(clean_ma.iloc[-1])
                ema_prev = float(clean_ema.iloc[-2])
                ma_prev = float(clean_ma.iloc[-2])
                
                fmt = ".4f" if ema_now < 1.0 else ".2f"
                
                if ema_now > ma_now and ema_prev <= ma_prev:
                    signals.append(f"📈 EMA10 ({ema_now:{fmt}}) > MA31 ({ma_now:{fmt}}) (CROSSOVER UP)")
                    ema_ma_score = 1.0
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append(f"📉 MA31 ({ma_now:{fmt}}) > EMA10 ({ema_now:{fmt}}) (CROSSOVER DOWN)")
                    ema_ma_score = 0.0
                elif ema_now > ma_now:
                    signals.append(f"🟢 EMA10 ({ema_now:{fmt}}) sopra MA31 ({ma_now:{fmt}})")
                    ema_ma_score = 0.75
                else:
                    signals.append(f"🔴 MA31 ({ma_now:{fmt}}) sopra EMA10 ({ema_now:{fmt}})")
                    ema_ma_score = 0.25

        # 2. STIMA TREND 7 GIORNI (PESO 15%)
        if len(close) >= 10:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
            extra_data = {
                'var_percent': var_percent,
                'target_price': target_price,
                'stop_loss': stop_loss
            }
            signals.append(format_trend_line(var_percent, target_price, stop_loss))
            
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

        # 3. DELTA % EMA10 vs MA31 CALIBRATO SU MEDIA 3 MESI (PESO 15%)
        if ema_series is not None and ma_series is not None:
            # Calcolo serie storica del Delta %
            delta_series = ((ema_series - ma_series) / ma_series) * 100.0
            curr_delta = float(delta_series.iloc[-1])
            
            # Media dei valori assoluti dei Delta negli ultimi 3 mesi (63 sessioni)
            historical_deltas = delta_series.tail(63).abs()
            avg_delta_3m = float(historical_deltas.mean()) if len(historical_deltas) > 0 else 1.0
            
            ratio_delta = curr_delta / avg_delta_3m if avg_delta_3m > 0 else 0.0
            sign = "+" if curr_delta > 0 else ""
            
            signals.append(f"📐 Delta EMA/MA: {sign}{curr_delta:.2f}% ({abs(ratio_delta):.1f}x media 3M: {avg_delta_3m:.2f}%)")
            
            if curr_delta > 0:
                if ratio_delta >= 1.5:
                    ema_ma_delta_score = 1.0
                elif ratio_delta >= 1.0:
                    ema_ma_delta_score = 0.85
                elif ratio_delta >= 0.5:
                    ema_ma_delta_score = 0.70
                else:
                    ema_ma_delta_score = 0.55
            else:
                if ratio_delta <= -1.5:
                    ema_ma_delta_score = 0.0
                elif ratio_delta <= -1.0:
                    ema_ma_delta_score = 0.15
                elif ratio_delta <= -0.5:
                    ema_ma_delta_score = 0.30
                else:
                    ema_ma_delta_score = 0.45

        # HEIKIN ASHI ANALYSIS (PARAMETRI 4 E 5)
        ha = calculate_heikin_ashi(df)
        if len(ha) >= 63:
            last_ha_close = float(ha['HA_Close'].iloc[-1])
            last_ha_open = float(ha['HA_Open'].iloc[-1])
            last_ha_low = float(ha['HA_Low'].iloc[-1])
            last_ha_high = float(ha['HA_High'].iloc[-1])
            
            # 4. HEIKIN ASHI - FORZA / ESTENSIONE CORPO VS MEDIA 3 MESI (PESO 15%)
            ha_bodies = (ha['HA_Close'] - ha['HA_Open']).abs()
            curr_body = float(ha_bodies.iloc[-1])
            avg_body_3m = float(ha_bodies.tail(63).mean())
            ratio_body = (curr_body / avg_body_3m) if avg_body_3m > 0 else 1.0
            
            is_green = last_ha_close > last_ha_open
            
            if is_green:
                if ratio_body >= 1.5:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Alta estensione)")
                    ha_force_score = 1.0
                elif ratio_body >= 1.0:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Sopra la media)")
                    ha_force_score = 0.80
                elif ratio_body >= 0.5:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Moderata)")
                    ha_force_score = 0.60
                else:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Corpo ridotto)")
                    ha_force_score = 0.50
            else:
                if ratio_body >= 1.5:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Forte pressione ribassista)")
                    ha_force_score = 0.0
                elif ratio_body >= 1.0:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Ribasso sostenuto)")
                    ha_force_score = 0.20
                elif ratio_body >= 0.5:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Ribasso contenuto)")
                    ha_force_score = 0.40
                else:
                    signals.append(f"🕯️ Espansione HA: {ratio_body:.1f}x media 3M (Corpo ridotto)")
                    ha_force_score = 0.50

            # 5. HEIKIN ASHI - STATO / COLORE & OMBRE (ANALISI TECNICA) (PESO 10%)
            upper_shadow = last_ha_high - max(last_ha_close, last_ha_open)
            lower_shadow = min(last_ha_close, last_ha_open) - last_ha_low
            total_range = last_ha_high - last_ha_low
            
            # Presenza ombra trascurabile / nulla (soglia rispetto alla chiusura)
            threshold = last_ha_close * 1e-4
            has_no_lower_shadow = lower_shadow <= threshold
            has_no_upper_shadow = upper_shadow <= threshold
            
            # Condizione di incertezza (Doji / Spinning Top): corpo piccolo e ombre presenti
            is_indecision = (curr_body / total_range < 0.25) if total_range > 0 else False
            
            if is_indecision:
                signals.append("⚪ HEIKIN ASHI: DOJI / INCERTEZZA (Ombre presenti, corpo stretto)")
                ha_state_score = 0.50
            elif is_green:
                if has_no_lower_shadow:
                    signals.append("🟢 HEIKIN ASHI: VERDE FORTE (Senza ombra inferiore)")
                    ha_state_score = 1.0
                else:
                    signals.append("🟢 HEIKIN ASHI: BARRA VERDE (Rialzista con ombra inf.)")
                    ha_state_score = 0.75
            else:
                if has_no_upper_shadow:
                    signals.append("🔴 HEIKIN ASHI: ROSSA FORTE (Senza ombra superiore)")
                    ha_state_score = 0.0
                else:
                    signals.append("🔴 HEIKIN ASHI: BARRA ROSSA (Ribassista con ombra sup.)")
                    ha_state_score = 0.25

        # 6. VOLUME VS MEDIA 3 MESI (~63 SESSIONI) (PESO 5%)
        if len(volume) >= 63:
            avg_vol_3m = float(volume.tail(63).mean())
            curr_vol = float(volume.iloc[-1])
            diff_pct = ((curr_vol - avg_vol_3m) / avg_vol_3m * 100.0) if avg_vol_3m > 0 else 0.0
            
            if curr_vol > avg_vol_3m * 1.5:
                signals.append(f"📊 Volume: +{diff_pct:.0f}% vs media 3 mesi")
                vol_score = 1.0
            elif curr_vol >= avg_vol_3m:
                signals.append(f"📊 Volume: nella media 3 mesi ({diff_pct:+.0f}%)")
                vol_score = 0.75
            else:
                signals.append(f"📊 Volume: sotto media 3 mesi ({diff_pct:.0f}%)")
                vol_score = 0.35

        # 7. CHIUSURA VS PRECEDENTE (PESO 5%)
        if len(close) >= 2:
            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            pct_change = ((last_close - prev_close) / prev_close) * 100.0
            sign = "+" if pct_change > 0 else ""
            
            signals.append(f"🔹 Chiusura vs Prec: {sign}{pct_change:.2f}%")
            
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

        # 8. ZIGZAG (PESO 5%)
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

        # 9. RSI 14 (PESO 5%)
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

        # 10. MACD 12,26,9 (PESO 5%)
        if len(close) >= 35:
            macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd_obj.macd().dropna()
            signal_line = macd_obj.macd_signal().dropna()
            
            if len(macd_line) > 1 and len(signal_line) > 1:
                m_now, s_now = float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
                m_prev, s_prev = float(macd_line.iloc[-2]), float(signal_line.iloc[-2])
                
                fmt = ".4f" if abs(m_now) < 1.0 else ".2f"
                
                if m_now > s_now and m_prev <= s_prev:
                    signals.append(f"📈 MACD ({m_now:{fmt}}) > Signal ({s_now:{fmt}}) (CROSSOVER UP)")
                    macd_score = 1.0
                elif m_now < s_now and m_prev >= s_prev:
                    signals.append(f"📉 MACD ({m_now:{fmt}}) < Signal ({s_now:{fmt}}) (CROSSOVER DOWN)")
                    macd_score = 0.0
                elif m_now > s_now:
                    signals.append(f"🟢 MACD ({m_now:{fmt}}) sopra Signal ({s_now:{fmt}})")
                    macd_score = 0.75
                else:
                    signals.append(f"🔴 MACD ({m_now:{fmt}}) sotto Signal ({s_now:{fmt}})")
                    macd_score = 0.25

        # SCORE FINALE BILANCIATO AL 100%
        final_score = (
            (ema_ma_score * 0.20) +
            (trend_score * 0.15) +
            (ema_ma_delta_score * 0.15) +
            (ha_force_score * 0.15) +
            (ha_state_score * 0.10) +
            (vol_score * 0.05) +
            (close_change_score * 0.05) +
            (zigzag_score * 0.05) +
            (rsi_score * 0.05) +
            (macd_score * 0.05)
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
            
        lines.append("----------------------------")
        
    return "\n".join(lines)

def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:
    """
    Invia un messaggio Telegram dividendolo automaticamente se supera la lunghezza massima.
    """
    MAX_LENGTH = 3800  # Soglia di sicurezza sotto il limite Telegram di 4096
    
    chunks = []
    if len(message) > MAX_LENGTH:
        lines = message.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) + 1 > MAX_LENGTH:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = len(line)
            else:
                current_chunk.append(line)
                current_length += len(line) + 1
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
    else:
        chunks = [message]

    success = True
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown" if use_markdown else None,
            "disable_web_page_preview": True
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                print(f"❌ Errore API Telegram ({resp.status_code}): {resp.text}")
                success = False
        except Exception as e:
            print(f"❌ Errore invio Telegram: {e}")
            success = False
        time.sleep(1) # Pausa tra un blocco e l'altro
        
    return success

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
                print("\n📩 Invio report Portafoglio...")
                send_telegram_message(token, chat_id, create_portfolio_daily_report(portfolio_results, descriptions))
                time.sleep(2)
            if watchlist_results:
                print("\n📩 Invio report Watchlist...")
                send_telegram_message(token, chat_id, create_watchlist_daily_report(watchlist_results, descriptions))
                
        print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")

if __name__ == "__main__":
    main()

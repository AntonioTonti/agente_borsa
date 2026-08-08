#!/usr/bin/env python3
"""
Agente di Trading - Analisi Settimanale (CHIUSURA)
Format Telegram: Riga singola per ticker con Delta % settimanale e Score.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta

sys.path.append('.')
from config import load_titoli_csv
from analysis_utils import (
    calculate_heikin_ashi,
    get_bullet,
    calculate_trend_estimate,
    format_trend_line
)

WEEKLY_PERIOD = "1y"
WEEKLY_INTERVAL = "1wk"
WEEKLY_MIN_POINTS = 20


def analyze_weekly_ticker(ticker: str) -> Tuple[List[str], float, Dict]:
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
        
        df = df.dropna()
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Variazione Settimanale
        if len(close) >= 2:
            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            extra_data['weekly_var_pct'] = ((last_close - prev_close) / prev_close) * 100.0

        # Stima Trend (3 settimane)
        if len(close) >= 6:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=3)
            extra_data.update({
                'var_percent': var_percent,
                'target_price': target_price,
                'stop_loss': stop_loss
            })
            signals.append(format_trend_line(var_percent, target_price, stop_loss))
        
        # 1. Heikin Ashi
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

        # 2. EMA10 vs MA31
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
                fmt = ".4f" if ema_now < 1.0 else ".2f"
                
                if ema_now > ma_now and ema_prev <= ma_prev:
                    signals.append(f"📈 EMA10 ({ema_now:{fmt}}) > MA31 ({ma_now:{fmt}}) (CROSSOVER UP)")
                    score += 0.25
                elif ma_now > ema_now and ma_prev <= ema_prev:
                    signals.append(f"📉 MA31 ({ma_now:{fmt}}) > EMA10 ({ema_now:{fmt}}) (CROSSOVER DOWN)")
                    score -= 0.25
                elif ema_now > ma_now:
                    signals.append(f"🟢 EMA10 ({ema_now:{fmt}}) sopra MA31 ({ma_now:{fmt}})")
                    score += 0.15
                else:
                    signals.append(f"🔴 MA31 ({ma_now:{fmt}}) sopra EMA10 ({ema_now:{fmt}})")
                    score -= 0.15

        # 3. RSI
        if len(close) >= 15:
            rsi = ta.momentum.rsi(close, window=14).dropna()
            if not rsi.empty:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    signals.append(f"⚠️ RSI: {rsi_val:.1f} (IPERCOMPRATO)")
                    score -= 0.15
                elif rsi_val < 30:
                    signals.append(f"⚠️ RSI: {rsi_val:.1f} (IPERVENDUTO)")
                    score += 0.10
                elif rsi_val > 60:
                    signals.append(f"📊 RSI: {rsi_val:.1f} (Zona Alta)")
                    score += 0.05
                elif rsi_val < 40:
                    signals.append(f"📊 RSI: {rsi_val:.1f} (Zona Bassa)")
                    score -= 0.05
                else:
                    signals.append(f"📊 RSI: {rsi_val:.1f} (Neutro)")

        # 4. Volume
        if len(volume) >= 10:
            avg_volume = float(volume.tail(10).mean())
            current_volume = float(volume.iloc[-1])
            if current_volume > avg_volume * 1.5:
                signals.append("📊 Volume +50% vs media 10 sett.")
                score += 0.10
            elif current_volume < avg_volume * 0.5:
                signals.append("📊 Volume sotto media 10 sett.")
                score -= 0.05

        # Score Finale
        other_indicators_score = score
        ha_normalized = (ha_color_score + 0.45) / 0.9
        final_score = (ha_normalized * 0.35) + (other_indicators_score * 0.65)
        final_score = max(0.0, min(1.0, final_score))
        
        return signals, round(final_score, 3), extra_data
        
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return signals, 0.5, extra_data


def create_weekly_report_section(title: str, results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    if not results:
        return f"{title}\nNessun dato disponibile."
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = [f"{title}\n"]
    
    for ticker, _, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        var_pct = extra_data.get('weekly_var_pct', 0.0)
        sign = "+" if var_pct > 0 else ""
        
        line = f"{bullet} {ticker} - {desc} {sign}{var_pct:.2f}% (score: {score:.3f})"
        lines.append(line)
        
    return "\n".join(lines)


def create_portfolio_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    return create_weekly_report_section("💰 *PORTAFOGLIO SETTIMANALE*", results, descriptions)


def create_watchlist_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    return create_weekly_report_section("👁️ *OSSERVATI SETTIMANALI*", results, descriptions)


def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
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
        print("📊 AGENTE DI TRADING - ANALISI SETTIMANALE (CHIUSURA)")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        
        portfolio_results = []
        if portfolio:
            print("\n💰 ANALISI PORTAFOGLIO")
            for ticker in portfolio:
                signals, score, extra_data = analyze_weekly_ticker(ticker)
                portfolio_results.append((ticker, signals, score, extra_data))
                
        watchlist_results = []
        if watchlist:
            print("\n👁️ ANALISI WATCHLIST")
            for ticker in watchlist:
                signals, score, extra_data = analyze_weekly_ticker(ticker)
                watchlist_results.append((ticker, signals, score, extra_data))
                
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            if portfolio_results:
                print("\n📩 Invio report Portafoglio...")
                send_telegram_message(token, chat_id, create_portfolio_report(portfolio_results, descriptions))
                time.sleep(2)
            if watchlist_results:
                print("\n📩 Invio report Watchlist...")
                send_telegram_message(token, chat_id, create_watchlist_report(watchlist_results, descriptions))
                
        print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")


if __name__ == "__main__":
    main()

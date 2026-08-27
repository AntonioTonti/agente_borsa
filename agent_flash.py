#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera (FLASH)
Format Telegram: Riga singola per ticker con Delta % giornaliera e Score al 100%.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta

sys.path.append('.')
from config import load_titoli_csv, DAILY_MIN_POINTS
from analysis_utils import (
    calculate_heikin_ashi,
    get_bullet,
    calculate_trend_estimate,
    format_trend_line
)
from web_generator import generate_web_page


def calculate_zigzag_trend(df: pd.DataFrame, deviation_pct: float = 5.0) -> int:
    if len(df) < 20:
        return 0
    highs = df['High'].values
    lows = df['Low'].values
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


def get_analyst_rating_score(tk: yf.Ticker) -> Tuple[float, str]:
    try:
        info = tk.info
        recommendation = info.get('recommendationKey', '').lower()
        
        if recommendation in ['strong_buy', 'buy']:
            return 1.0, f"Rating Analisti: {recommendation.upper()} 🟢"
        elif recommendation in ['outperform', 'overweight']:
            return 0.75, f"Rating Analisti: {recommendation.upper()} 🟢"
        elif recommendation in ['hold', 'neutral']:
            return 0.50, f"Rating Analisti: {recommendation.upper()} ⚪"
        elif recommendation in ['underperform', 'underweight']:
            return 0.25, f"Rating Analisti: {recommendation.upper()} 🔴"
        elif recommendation in ['sell', 'strong_sell']:
            return 0.0, f"Rating Analisti: {recommendation.upper()} 🔴"
        else:
            return 0.50, "Rating Analisti: N/D ⚪"
    except Exception:
        return 0.50, "Rating Analisti: N/D ⚪"


def analyze_daily_ticker(ticker: str) -> Tuple[List[str], float, Dict, Optional[pd.DataFrame]]:
    signals = []
    extra_data = {'daily_var_pct': 0.0}
    
    ema_ma_score = 0.5
    trend_score = 0.5
    analyst_score = 0.5
    ema_ma_delta_score = 0.5
    ha_force_score = 0.5
    ha_state_score = 0.5
    zigzag_score = 0.5
    vol_score = 0.5
    close_change_score = 0.5
    rsi_score = 0.5
    macd_score = 0.5

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="6mo", interval="1d", auto_adjust=True)
        
        if df.empty or len(df) < DAILY_MIN_POINTS:
            print(f"⚠️ {ticker}: Dati vuoti o insufficienti ({len(df)} righe).")
            return signals, 0.5, extra_data, None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # --- FIX RIGOROSO PREZZO E VARIAZIONE INTRADAY ---
        fast_info = getattr(tk, 'fast_info', {})
        last_price = fast_info.get('lastPrice', None)
        prev_close = fast_info.get('previousClose', None)

        if last_price is None or np.isnan(last_price):
            last_price = float(df['Close'].iloc[-1])

        # Controllo basato sulle date del DataFrame
        today_date = datetime.now().date()
        last_df_date = df.index[-1].date()

        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            if last_df_date == today_date and len(df) >= 2:
                prev_close = float(df['Close'].iloc[-2])
            elif last_df_date < today_date and len(df) >= 1:
                prev_close = float(df['Close'].iloc[-1])
            else:
                prev_close = last_price

        # Sostituiamo il prezzo corrente nell'ultima riga se la sessione è di oggi
        if last_df_date == today_date:
            df.iloc[-1, df.columns.get_loc('Close')] = last_price

        pct_change = ((last_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        extra_data['daily_var_pct'] = pct_change
        # ------------------------------------------------

        close = df['Close']
        volume = df['Volume']

        # 1. EMA10 vs MA31 (15%)
        clean_ema, clean_ma = None, None
        if len(close) >= 32:
            ema10 = ta.trend.ema_indicator(close, window=10)
            ma31 = ta.trend.sma_indicator(close, window=31)
            clean_ema = ema10.dropna()
            clean_ma = ma31.dropna()
            
            if len(clean_ema) > 1 and len(clean_ma) > 1:
                ema_now, ma_now = float(clean_ema.iloc[-1]), float(clean_ma.iloc[-1])
                ema_prev, ma_prev = float(clean_ema.iloc[-2]), float(clean_ma.iloc[-2])
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

        # 2. STIMA TREND 7 GIORNI (13%)
        if len(close) >= 10:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
            extra_data.update({'var_percent': var_percent, 'target_price': target_price, 'stop_loss': stop_loss})
            signals.append(format_trend_line(var_percent, target_price, stop_loss))
            
            if var_percent > 3.0: trend_score = 1.0
            elif var_percent > 0.0: trend_score = 0.75
            elif var_percent == 0.0: trend_score = 0.50
            elif var_percent > -3.0: trend_score = 0.25
            else: trend_score = 0.0

        # 3. RATING ANALISTI (5%)
        analyst_score, analyst_msg = get_analyst_rating_score(tk)
        signals.append(f"🎯 {analyst_msg}")

        # 4. DELTA % EMA10/MA31 (12%)
        if clean_ema is not None and clean_ma is not None and len(clean_ma) >= 63:
            common_idx = clean_ema.index.intersection(clean_ma.index)
            delta_series = ((clean_ema.loc[common_idx] - clean_ma.loc[common_idx]) / clean_ma.loc[common_idx]) * 100.0
            curr_delta = float(delta_series.iloc[-1])
            avg_delta_3m = float(delta_series.tail(63).abs().mean())
            sign = "+" if curr_delta > 0 else ""
            signals.append(f"📐 Delta EMA10/MA31: {sign}{curr_delta:.2f}% (Media Abs 3M: {avg_delta_3m:.2f}%)")
            
            if curr_delta > 0:
                ema_ma_delta_score = 1.0 if (avg_delta_3m > 0 and curr_delta >= avg_delta_3m * 1.5) else (0.80 if curr_delta >= avg_delta_3m else 0.60)
            else:
                abs_curr = abs(curr_delta)
                ema_ma_delta_score = 0.0 if (avg_delta_3m > 0 and abs_curr >= avg_delta_3m * 1.5) else (0.20 if abs_curr >= avg_delta_3m else 0.40)

        # 5 & 6. HEIKIN ASHI (FORZA 15%, STATO 10%)
        ha = calculate_heikin_ashi(df)
        if len(ha) >= 63:
            last_ha_close = float(ha['HA_Close'].iloc[-1])
            last_ha_open = float(ha['HA_Open'].iloc[-1])
            last_ha_low = float(ha['HA_Low'].iloc[-1])
            last_ha_high = float(ha['HA_High'].iloc[-1])
            
            ha_body = abs(last_ha_close - last_ha_open)
            ha_range = max(1e-6, last_ha_high - last_ha_low)
            upper_shadow = last_ha_high - max(last_ha_open, last_ha_close)
            lower_shadow = min(last_ha_open, last_ha_close) - last_ha_low
            
            is_green = last_ha_close >= last_ha_open
            is_doji = (ha_body / ha_range) < 0.15
            
            ha_bodies = (ha['HA_Close'] - ha['HA_Open']).abs()
            avg_body_3m = float(ha_bodies.tail(63).mean())
            ratio_body = (ha_bodies.iloc[-1] / avg_body_3m) if avg_body_3m > 0 else 1.0
            
            if is_green:
                ha_force_score = 1.0 if ratio_body >= 1.5 else (0.75 if ratio_body >= 1.0 else 0.50)
            else:
                ha_force_score = 0.0 if ratio_body >= 1.5 else (0.25 if ratio_body >= 1.0 else 0.40)

            if is_doji:
                ha_state_score = 0.50
                ha_desc = "Doji (Incertezza)"
            elif is_green:
                if lower_shadow <= (ha_range * 0.03):
                    ha_state_score = 1.0
                    ha_desc = "Verde senza ombra inf. (Molto Forte 🟢)"
                elif upper_shadow > lower_shadow:
                    ha_state_score = 0.75
                    ha_desc = "Verde (Spinta Rialzista 🟢)"
                else:
                    ha_state_score = 0.60
                    ha_desc = "Verde con ombra inf. (Pressione di vendita)"
            else:
                if upper_shadow <= (ha_range * 0.03):
                    ha_state_score = 0.0
                    ha_desc = "Rossa senza ombra sup. (Molto Debole 🔴)"
                elif lower_shadow > upper_shadow:
                    ha_state_score = 0.25
                    ha_desc = "Rossa (Spinta Ribassista 🔴)"
                else:
                    ha_state_score = 0.40
                    ha_desc = "Rossa con ombra sup. (Pressione di acquisto)"

            signals.append(f"🕯️ Heikin Ashi: {ha_desc} - Forza Corpo: {ratio_body:.2f}x media 3M")

        # 7. ZIGZAG (10%)
        zz_trend = calculate_zigzag_trend(df, deviation_pct=5.0)
        zigzag_score = 1.0 if zz_trend == 1 else (0.0 if zz_trend == -1 else 0.5)
        zz_desc = "Rialzista 🟢" if zz_trend == 1 else ("Ribassista 🔴" if zz_trend == -1 else "Neutro ⚪")
        signals.append(f"⚡ ZigZag (5%): Trend {zz_desc}")

        # 8. VOLUME (5%)
        if len(volume) >= 63:
            avg_vol_3m = float(volume.tail(63).mean())
            curr_vol = float(volume.iloc[-1])
            vol_ratio = (curr_vol / avg_vol_3m) if avg_vol_3m > 0 else 1.0
            
            if curr_vol > avg_vol_3m * 1.5:
                vol_score = 1.0
                vol_desc = "Volumi in forte aumento (>150% media) 🟢"
            elif curr_vol >= avg_vol_3m:
                vol_score = 0.75
                vol_desc = "Volumi sopra la media 3M 🟢"
            else:
                vol_score = 0.35
                vol_desc = "Volumi sotto la media 3M 🔴"
                
            signals.append(f"📊 Volumi: {curr_vol:,.0f} vs Media 3M {avg_vol_3m:,.0f} ({vol_desc})")

        # --- ESTRAZIONE PREZZO LIVE E PREVIOUS CLOSE ---
        fast_info = getattr(tk, 'fast_info', {})
        last_price = fast_info.get('lastPrice', None)
        prev_close = fast_info.get('previousClose', None)

        # Fallback nel caso in cui fast_info sia momentaneamente indisponibile
        if last_price is None or np.isnan(last_price):
            last_price = float(df['Close'].iloc[-1])
        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else last_price

        # Variazione % REALE e INTRADAY
        pct_change = ((last_price - prev_close) / prev_close) * 100.0
        extra_data['daily_var_pct'] = pct_change
        # -----------------------------------------------

        # 9. VARIAZIONE % GIORNALIERA / INTRADAY (5%)
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

        sign_chg = "+" if pct_change > 0 else ""
        # Dicitura aggiornata per chiarire che si tratta del dato live/intraday
        signals.append(f"💵 Variazione Intraday: {sign_chg}{pct_change:.2f}% (Prezzo: {last_price:.2f} | Chiusura Ieri: {prev_close:.2f})")

        # 10. RSI (5%)
        if len(close) >= 15:
            rsi = ta.momentum.rsi(close, window=14).dropna()
            if not rsi.empty:
                rsi_val = float(rsi.iloc[-1])
                if rsi_val > 70:
                    rsi_score = 0.15
                    rsi_desc = "Ipercomprato (>70) 🔴"
                elif rsi_val < 30:
                    rsi_score = 0.85
                    rsi_desc = "Ipervenduto (<30) 🟢"
                elif rsi_val > 60:
                    rsi_score = 0.65
                    rsi_desc = "Fasciatura Rialzista (60-70) 🟢"
                elif rsi_val < 40:
                    rsi_score = 0.35
                    rsi_desc = "Fasciatura Ribassista (30-40) 🔴"
                else:
                    rsi_score = 0.50
                    rsi_desc = "Zona Neutra (40-60) ⚪"
                
                signals.append(f"🟣 RSI (14): {rsi_val:.2f} - {rsi_desc}")

        # 11. MACD (5%)
        if len(close) >= 35:
            macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            m_line, s_line = macd_obj.macd().dropna(), macd_obj.macd_signal().dropna()
            if len(m_line) > 1 and len(s_line) > 1:
                m_now, s_now = float(m_line.iloc[-1]), float(s_line.iloc[-1])
                m_prev, s_prev = float(m_line.iloc[-2]), float(s_line.iloc[-2])
                
                if m_now > s_now and m_prev <= s_prev:
                    macd_score = 1.0
                    macd_desc = "Crossover Rialzista (CROSSOVER UP) 📈"
                elif m_now < s_now and m_prev >= s_prev:
                    macd_score = 0.0
                    macd_desc = "Crossover Ribassista (CROSSOVER DOWN) 📉"
                elif m_now > s_now:
                    macd_score = 0.75
                    macd_desc = "Sopra la Signal Line (Fase Positiva) 🟢"
                else:
                    macd_score = 0.25
                    macd_desc = "Sotto la Signal Line (Fase Negativa) 🔴"
                    
                signals.append(f"📊 MACD: {macd_desc}")

        # SCORE FINALE CON NUOVE PONDERAZIONI (100% TOTALE)
        final_score = (
            (ema_ma_score * 0.15) + 
            (trend_score * 0.13) + 
            (analyst_score * 0.05) + 
            (ema_ma_delta_score * 0.12) +
            (ha_force_score * 0.15) + 
            (ha_state_score * 0.10) + 
            (zigzag_score * 0.10) +
            (vol_score * 0.05) + 
            (close_change_score * 0.05) + 
            (rsi_score * 0.05) + 
            (macd_score * 0.05)
        )
        return signals, round(max(0.0, min(1.0, final_score)), 3), extra_data, df

    except Exception as e:
        print(f"❌ Errore durante l'analisi di {ticker}: {e}")
        return signals, 0.5, extra_data, None


def create_daily_report_section(title: str, results: List[Tuple[str, List[str], float, Dict, Optional[pd.DataFrame]]], descriptions: Dict) -> str:
    if not results:
        return f"{title}\nNessun dato disponibile."
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = [f"{title}\n"]
    
    for ticker, _, score, extra_data, _ in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        var_pct = extra_data.get('daily_var_pct', 0.0)
        sign = "+" if var_pct > 0 else ""
        
        url = f"https://antoniotonti.github.io/agente_borsa/flash/{ticker}.html"
        line = f"{bullet} [{ticker}]({url}) - {desc} {sign}{var_pct:.2f}% (score: {score:.3f})"
        lines.append(line)
        
    return "\n".join(lines)


def create_portfolio_daily_report(results: List[Tuple[str, List[str], float, Dict, Optional[pd.DataFrame]]], descriptions: Dict) -> str:
    return create_daily_report_section("💰 *PORTAFOGLIO GIORNALIERO*", results, descriptions)


def create_watchlist_daily_report(results: List[Tuple[str, List[str], float, Dict, Optional[pd.DataFrame]]], descriptions: Dict) -> str:
    return create_daily_report_section("👁️ *OSSERVATI GIORNALIERI*", results, descriptions)


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
        print("📊 AGENTE DI TRADING - ANALISI GIORNALIERA (FLASH)")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        
        portfolio_results = []
        if portfolio:
            print("\n💰 ANALISI PORTAFOGLIO")
            for ticker in portfolio:
                signals, score, extra_data, df = analyze_daily_ticker(ticker)
                portfolio_results.append((ticker, signals, score, extra_data, df))
                if df is not None and not df.empty:
                    desc = descriptions.get(ticker, ticker)
                    generate_web_page(ticker, desc, "flash", df, score, signals)
                time.sleep(0.5)
                
        watchlist_results = []
        if watchlist:
            print("\n👁️ ANALISI WATCHLIST")
            for ticker in watchlist:
                signals, score, extra_data, df = analyze_daily_ticker(ticker)
                watchlist_results.append((ticker, signals, score, extra_data, df))
                if df is not None and not df.empty:
                    desc = descriptions.get(ticker, ticker)
                    generate_web_page(ticker, desc, "flash", df, score, signals)
                time.sleep(0.5)
                
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

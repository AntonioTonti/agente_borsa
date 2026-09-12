#!/usr/bin/env python3
"""
Agente ETF - Analisi Multi-Timeframe (1H + 1D) per ETF a Leva / Direzionali
- Timeframe 1H: Operativo Intraday
- Timeframe 1D: Filtro di Trend di Fondo
- Integrazione Risk Management (ATR, SL, TP, Position Sizing)
- Integrazione State Management (Delta Score e Delta Variazione)
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
from analysis_utils import calculate_heikin_ashi, get_bullet
from web_generator import generate_web_page
from state_manager import load_previous_state, save_current_state, calculate_deltas
from risk_manager import calculate_atr, compute_risk_levels


def load_etf_from_csv(csv_path: str = "titoli.csv") -> Tuple[List[str], Dict[str, str]]:
    etf_tickers = []
    descriptions = {}
    
    if not os.path.exists(csv_path):
        print(f"❌ File {csv_path} non trovato.")
        return etf_tickers, descriptions

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        for _, row in df.iterrows():
            code = str(row['codice']).strip()
            tipo = str(row['tipo']).strip().upper()
            desc = str(row['descrizione']).strip()
            descriptions[code] = desc
            
            if tipo == 'ETF' or 'ETF' in desc.upper() or 'ETF' in code.upper() or '2X' in desc.upper():
                etf_tickers.append(code)
                
        print(f"✅ CSV caricato: trovati {len(etf_tickers)} titoli ETF")
    except Exception as e:
        print(f"❌ Errore lettura CSV: {e}")
        
    return etf_tickers, descriptions


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    atr = ta.volatility.average_true_range(high, low, close, window=period)
    hl2 = (high + low) / 2.0
    
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    upperband = pd.Series(0.0, index=df.index)
    lowerband = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)
    
    for i in range(1, len(df)):
        if basic_upperband.iloc[i] < upperband.iloc[i-1] or close.iloc[i-1] > upperband.iloc[i-1]:
            upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            upperband.iloc[i] = upperband.iloc[i-1]
            
        if basic_lowerband.iloc[i] > lowerband.iloc[i-1] or close.iloc[i-1] < lowerband.iloc[i-1]:
            lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            lowerband.iloc[i] = lowerband.iloc[i-1]
            
        if close.iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            elif direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
                
    st_line = pd.Series(np.where(direction == 1, lowerband, upperband), index=df.index)
    return st_line, direction


def compute_timeframe_score(df: pd.DataFrame, timeframe_label: str = "1H") -> Tuple[float, List[str]]:
    signals = []
    if df is None or len(df) < 15:
        return 0.5, ["⚠️ Dati insufficienti per l'analisi."]

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    close = df['Close']
    volume = df['Volume']

    st_score, macd_score, ha_score = 0.5, 0.5, 0.5
    rsi_score, ema_ma_score, vol_score = 0.5, 0.5, 0.5

    # 1. ADX (KILL-SWITCH)
    adx_val = 0.0
    is_lateral = True
    if len(df) >= 15:
        adx_df = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        adx_series = adx_df.adx().dropna()
        if not adx_series.empty:
            adx_val = float(adx_series.iloc[-1])
            is_lateral = adx_val < 20.0

    if is_lateral:
        signals.append(f"⛔ ADX ({timeframe_label}): {adx_val:.1f} - FASE LATERALE (Blocco) ⚠️")
    else:
        signals.append(f"⚡ ADX ({timeframe_label}): {adx_val:.1f} - Trend confermato 🟢")

    # 2. SUPERTREND
    st_line, st_dir = calculate_supertrend(df, period=10, multiplier=3.0)
    last_st_dir = st_dir.iloc[-1]
    last_st_val = st_line.iloc[-1]
    fmt = ".4f" if close.iloc[-1] < 1.0 else ".2f"
    
    if last_st_dir == 1:
        st_score = 1.0
        signals.append(f"🟢 Supertrend ({timeframe_label}): RIALZISTA (Supp: {last_st_val:{fmt}})")
    else:
        st_score = 0.0
        signals.append(f"🔴 Supertrend ({timeframe_label}): RIBASSISTA (Res: {last_st_val:{fmt}})")

    # 3. MACD
    macd_obj = ta.trend.MACD(close=close, window_slow=17, window_fast=8, window_sign=9)
    m_line, s_line = macd_obj.macd().dropna(), macd_obj.macd_signal().dropna()
    if len(m_line) > 1 and len(s_line) > 1:
        m_now, s_now = float(m_line.iloc[-1]), float(s_line.iloc[-1])
        m_prev, s_prev = float(m_line.iloc[-2]), float(s_line.iloc[-2])
        if m_now > s_now and m_prev <= s_prev:
            macd_score = 1.0
            signals.append(f"📈 MACD ({timeframe_label}): CROSSOVER RIALZISTA 🟢")
        elif m_now < s_now and m_prev >= s_prev:
            macd_score = 0.0
            signals.append(f"📉 MACD ({timeframe_label}): CROSSOVER RIBASSISTA 🔴")
        elif m_now > s_now:
            macd_score = 0.75
            signals.append(f"🟢 MACD ({timeframe_label}): Positivo")
        else:
            macd_score = 0.25
            signals.append(f"🔴 MACD ({timeframe_label}): Negativo")

    # 4. HEIKIN ASHI
    ha = calculate_heikin_ashi(df)
    if len(ha) >= 5:
        last_ha_close = float(ha['HA_Close'].iloc[-1])
        last_ha_open = float(ha['HA_Open'].iloc[-1])
        last_ha_low = float(ha['HA_Low'].iloc[-1])
        last_ha_high = float(ha['HA_High'].iloc[-1])
        ha_range = max(1e-6, last_ha_high - last_ha_low)
        upper_shadow = last_ha_high - max(last_ha_open, last_ha_close)
        lower_shadow = min(last_ha_open, last_ha_close) - last_ha_low
        is_green = last_ha_close >= last_ha_open
        
        if is_green:
            if lower_shadow <= (ha_range * 0.03):
                ha_score = 1.0
            else:
                ha_score = 0.70
        else:
            if upper_shadow <= (ha_range * 0.03):
                ha_score = 0.0
            else:
                ha_score = 0.30

    # 5. RSI (9)
    rsi = ta.momentum.rsi(close, window=9).dropna()
    if not rsi.empty:
        rsi_val = float(rsi.iloc[-1])
        if rsi_val > 70: rsi_score = 0.30
        elif rsi_val < 30: rsi_score = 0.80
        elif rsi_val >= 50: rsi_score = 0.85
        else: rsi_score = 0.20

    # 6. EMA10 vs MA31
    if len(close) >= 31:
        ema10 = ta.trend.ema_indicator(close, window=10).iloc[-1]
        ma31 = ta.trend.sma_indicator(close, window=31).iloc[-1]
        diff_pct = ((ema10 - ma31) / ma31) * 100.0
        if diff_pct > 0.1: ema_ma_score = 0.85
        elif diff_pct < -0.1: ema_ma_score = 0.15
        else: ema_ma_score = 0.50

    # 7. VOLUMI
    if len(volume) >= 10:
        avg_vol = float(volume.tail(10).mean())
        curr_vol = float(volume.iloc[-1])
        if curr_vol >= avg_vol: vol_score = 0.85
        else: vol_score = 0.35

    raw_score = (
        (st_score * 0.20) + (macd_score * 0.20) + (ha_score * 0.20) +
        (rsi_score * 0.15) + (ema_ma_score * 0.15) + (vol_score * 0.10)
    )

    final_score = min(0.45, raw_score * 0.50) if is_lateral else raw_score
    return round(max(0.0, min(1.0, final_score)), 3), signals


def analyze_etf(ticker: str, prev_state: Dict) -> Dict:
    result = {
        'ticker': ticker, 'score_1h': 0.5, 'score_1d': 0.5,
        'signals_1h': [], 'signals_1d': [], 'daily_var_pct': 0.0,
        'df_1h': None, 'delta_score': 0.0, 'delta_var': 0.0,
        'risk': {}
    }

    try:
        tk = yf.Ticker(ticker)
        df_1h = tk.history(period="12d", interval="1h", auto_adjust=True)
        if df_1h.empty or len(df_1h) < 15:
            df_1h = tk.history(period="1mo", interval="1d", auto_adjust=True)
            
        result['df_1h'] = df_1h
        df_1d = tk.history(period="6mo", interval="1d", auto_adjust=True)

        score_1h, signals_1h = compute_timeframe_score(df_1h, "1H")
        score_1d, signals_1d = compute_timeframe_score(df_1d, "1D")
        
        result['score_1h'] = score_1h
        result['score_1d'] = score_1d

        # Calcolo Variazione e Deltas (State Manager)
        last_price = 0.0
        pct_change = 0.0
        try:
            fast_info = getattr(tk, 'fast_info', {})
            last_price = fast_info.get('lastPrice', None)
            prev_close = fast_info.get('previousClose', None)
            
            if last_price is None or np.isnan(last_price):
                if df_1h is not None and not df_1h.empty:
                    last_price = float(df_1h['Close'].iloc[-1])

            if prev_close and prev_close > 0:
                pct_change = ((last_price - prev_close) / prev_close) * 100.0
        except Exception:
            pass

        result['daily_var_pct'] = pct_change
        
        # State Manager: Delta
        delta_var, delta_score = calculate_deltas(ticker, pct_change, score_1h, prev_state)
        result['delta_var'] = delta_var
        result['delta_score'] = delta_score

        # Gestione del Rischio (Risk Manager)
        atr_1h = calculate_atr(df_1h, 14)
        direction = "BULLISH" if score_1h >= 0.6 else "BEARISH" if score_1h <= 0.4 else "NEUTRAL"
        sl, tp, risk_label, sizing = compute_risk_levels(last_price, atr_1h, direction)
        
        signals_1h.append(f"🛡️ Rischio: {risk_label} | Size: {sizing}% | SL: {sl:.2f} | TP: {tp:.2f}")
        
        result['signals_1h'] = signals_1h
        result['signals_1d'] = signals_1d
        result['risk'] = {'label': risk_label, 'sizing': sizing, 'sl': sl, 'tp': tp}

    except Exception as e:
        print(f"❌ Errore ETF {ticker}: {e}")

    return result


def create_unified_report(results: List[Dict], descriptions: Dict) -> str:
    if not results:
        return "📊 *AGENTE ETF - REPORT ORARIO*\nNessun ETF disponibile."
    
    sorted_results = sorted(results, key=lambda x: x['score_1h'], reverse=True)
    now_str = datetime.now().strftime('%H:%M')
    lines = [f"📊 *AGENTE ETF LEVA - MONITORAGGIO ({now_str})*\n"]
    
    for res in sorted_results:
        ticker = res['ticker']
        desc = descriptions.get(ticker, ticker)
        s_1h = res['score_1h']
        s_1d = res['score_1d']
        b_1h = get_bullet(s_1h)
        
        d_score = f"{res['delta_score']:+.3f}"
        var_pct = res['daily_var_pct']
        sign = "+" if var_pct > 0 else ""
        
        url = f"https://antoniotonti.github.io/agente_borsa/flash/{ticker}.html"
        risk_info = res.get('risk', {})
        
        line = (
            f"🔹 [{ticker}]({url}) - *{desc}* (Oggi: {sign}{var_pct:.2f}%)\n"
            f"   ├ ⚡ *1H:* {b_1h} `{s_1h:.3f}` (Δ: {d_score})\n"
            f"   ├ 🛡️ *Rischio:* {risk_info.get('label','-')} | Size: {risk_info.get('sizing',0)}%\n"
            f"   └ 🎯 *SL:* {risk_info.get('sl',0):.2f} | *TP:* {risk_info.get('tp',0):.2f}"
        )
        
        if s_1h >= 0.70 and s_1d >= 0.70:
            line += "\n   🔥 *CONFLUENZA RIALZISTA (1H + 1D)* 🟢"
            
        lines.append(line + "\n")
        
    return "\n".join(lines)


def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=15)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    start_time = time.time()
    try:
        print("📊 AGENTE ETF - RISK & STATE MANAGER INTEGRATI")
        etf_tickers, descriptions = load_etf_from_csv("titoli.csv")
        
        prev_state = load_previous_state("ETF")
        new_state = {}
        etf_results = []
        
        for ticker in etf_tickers:
            print(f"-> Analisi: {ticker}")
            res = analyze_etf(ticker, prev_state)
            etf_results.append(res)
            
            # Aggiornamento stato
            new_state[ticker] = {
                "var_pct": res['daily_var_pct'],
                "score": res['score_1h']
            }
            
            if res['df_1h'] is not None:
                combined_signals = ["--- TIMEFRAME 1H ---"] + res['signals_1h'] + ["", "--- TIMEFRAME 1D ---"] + res['signals_1d']
                generate_web_page(ticker, descriptions.get(ticker, ticker), "flash", res['df_1h'], res['score_1h'], combined_signals)
                
            time.sleep(0.3)
            
        # Salva stato aggiornato a fine elaborazione
        save_current_state("ETF", new_state)
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if token and chat_id and etf_results:
            unified_report = create_unified_report(etf_results, descriptions)
            send_telegram_message(token, chat_id, unified_report)
                
        print(f"🏁 Completato in {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")

if __name__ == "__main__":
    main()

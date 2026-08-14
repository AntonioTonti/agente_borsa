#!/usr/bin/env python3
"""
Agente ETF - Analisi Oraria per ETF a Leva / Direzionali (Versione Corretta Anti-Falso Positivo)
- Timeframe: Orario (1h)
- ADX usato come KILL-SWITCH Anti-Lateralità (se ADX < 20 lo score viene penalizzato drasticamente)
- Zoom grafico ridotto a 10gg per leggibilità candele orarie
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
from analysis_utils import calculate_heikin_ashi, get_bullet, calculate_trend_estimate, format_trend_line
from web_generator import generate_web_page

def load_etf_from_csv(csv_path: str = "titoli.csv") -> Tuple[List[str], Dict[str, str]]:
    """Carica i titoli dal CSV leggendo colonna 'tipo' == 'ETF' o presenza keywords."""
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
                
        print(f"✅ CSV caricato: trovati {len(etf_tickers)} titoli ETF ({', '.join(etf_tickers)})")
    except Exception as e:
        print(f"❌ Errore lettura CSV: {e}")
        
    return etf_tickers, descriptions


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Calcola il Supertrend per individuare la direzione del trend intraday."""
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


def analyze_etf_leveraged(ticker: str) -> Tuple[List[str], float, Dict, Optional[pd.DataFrame]]:
    """Analisi specifica per ETF a leva su timeframe orario con Filtro Anti-Lateralità."""
    signals = []
    extra_data = {'daily_var_pct': 0.0}
    
    st_score = 0.5        # 20%
    macd_score = 0.5      # 20%
    ha_score = 0.5        # 20%
    rsi_score = 0.5       # 15%
    ema_ma_score = 0.5   # 15%
    vol_score = 0.5      # 10%

    try:
        tk = yf.Ticker(ticker)
        # Scarichiamo dati orari degli ultimi 12 giorni per un grafico pulito
        df = tk.history(period="12d", interval="1h", auto_adjust=True)
        
        if df.empty or len(df) < 20:
            df = tk.history(period="1mo", interval="1d", auto_adjust=True)
            
        if df.empty or len(df) < 15:
            print(f"⚠️ {ticker}: Dati insufficienti ({len(df)} righe).")
            return signals, 0.5, extra_data, None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        close = df['Close']
        volume = df['Volume']

        # 1. FILTRO FONDAMENTALE ADX (KILL-SWITCH ANTI-LATERALITÀ)
        adx_val = 0.0
        is_lateral = True
        if len(df) >= 15:
            adx_df = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            adx_series = adx_df.adx().dropna()
            if not adx_series.empty:
                adx_val = float(adx_series.iloc[-1])
                is_lateral = adx_val < 20.0

        if is_lateral:
            signals.append(f"⛔ ADX: {adx_val:.1f} - FASE LATERALE DETETTATA (Blocco segnale d'acquisto) ⚠️")
        else:
            signals.append(f"⚡ ADX: {adx_val:.1f} - Trend in corso confermato 🟢")

        # 2. SUPERTREND ORARIO (20%)
        st_line, st_dir = calculate_supertrend(df, period=10, multiplier=3.0)
        last_st_dir = st_dir.iloc[-1]
        last_st_val = st_line.iloc[-1]
        fmt = ".4f" if close.iloc[-1] < 1.0 else ".2f"
        
        if last_st_dir == 1:
            st_score = 1.0
            signals.append(f"🟢 Supertrend ORARIO: RIALZISTA (Supporto: {last_st_val:{fmt}})")
        else:
            st_score = 0.0
            signals.append(f"🔴 Supertrend ORARIO: RIBASSISTA (Resistenza: {last_st_val:{fmt}})")

        # 3. MACD VELOCE (8, 17, 9) (20%)
        macd_obj = ta.trend.MACD(close=close, window_slow=17, window_fast=8, window_sign=9)
        m_line, s_line = macd_obj.macd().dropna(), macd_obj.macd_signal().dropna()
        if len(m_line) > 1 and len(s_line) > 1:
            m_now, s_now = float(m_line.iloc[-1]), float(s_line.iloc[-1])
            m_prev, s_prev = float(m_line.iloc[-2]), float(s_line.iloc[-2])
            
            if m_now > s_now and m_prev <= s_prev:
                macd_score = 1.0
                signals.append("📈 MACD Veloce: CROSSOVER RIALZISTA 🟢")
            elif m_now < s_now and m_prev >= s_prev:
                macd_score = 0.0
                signals.append("📉 MACD Veloce: CROSSOVER RIBASSISTA 🔴")
            elif m_now > s_now:
                macd_score = 0.75
                signals.append("🟢 MACD Veloce: Positivo (Sopra Signal Line)")
            else:
                macd_score = 0.25
                signals.append("🔴 MACD Veloce: Negativo (Sotto Signal Line)")

        # 4. HEIKIN ASHI CANDLE ANALYSIS (20%)
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
                    signals.append("🕯️ Heikin Ashi: Forte Spinta Verde (Senza Ombra Inf.) 🟢")
                else:
                    ha_score = 0.70
                    signals.append("🕯️ Heikin Ashi: Candela Verde 🟢")
            else:
                if upper_shadow <= (ha_range * 0.03):
                    ha_score = 0.0
                    signals.append("🕯️ Heikin Ashi: Forte Spinta Rossa (Senza Ombra Sup.) 🔴")
                else:
                    ha_score = 0.30
                    signals.append("🕯️ Heikin Ashi: Candela Rossa 🔴")

        # 5. RSI VELOCE (9) (15%)
        rsi = ta.momentum.rsi(close, window=9).dropna()
        if not rsi.empty:
            rsi_val = float(rsi.iloc[-1])
            if rsi_val > 70:
                rsi_score = 0.30
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Ipercomprato (Attenzione) ⚠️")
            elif rsi_val < 30:
                rsi_score = 0.80
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Ipervenduto (Rimbalzo possibile) 🟢")
            elif rsi_val >= 50:
                rsi_score = 0.85
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Zona di Forza 🟢")
            else:
                rsi_score = 0.20
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Zona di Debolezza 🔴")

        # 6. EMA10 vs MA31 (15%)
        if len(close) >= 31:
            ema10 = ta.trend.ema_indicator(close, window=10).iloc[-1]
            ma31 = ta.trend.sma_indicator(close, window=31).iloc[-1]
            # Verifica scarto % per evitare falsi incroci da 0.0001
            diff_pct = ((ema10 - ma31) / ma31) * 100.0
            
            if diff_pct > 0.1:
                ema_ma_score = 0.85
                signals.append(f"📊 Medie: EMA10 ({ema10:{fmt}}) sopra MA31 ({ma31:{fmt}}) (+{diff_pct:.2f}%) 🟢")
            elif diff_pct < -0.1:
                ema_ma_score = 0.15
                signals.append(f"📊 Medie: MA31 ({ma31:{fmt}}) sopra EMA10 ({ema10:{fmt}}) ({diff_pct:.2f}%) 🔴")
            else:
                ema_ma_score = 0.50
                signals.append(f"📊 Medie: EMA10 e MA31 Sovrapposte (Incertezza) ⚪")

        # 7. VOLUMI ULTIMA CANDELA (10%)
        if len(volume) >= 10:
            avg_vol = float(volume.tail(10).mean())
            curr_vol = float(volume.iloc[-1])
            if curr_vol >= avg_vol:
                vol_score = 0.85
                signals.append(f"📊 Volumi: Sopra la media oraria ({curr_vol:,.0f}) 🟢")
            else:
                vol_score = 0.35
                signals.append(f"📊 Volumi: Sotto la media oraria ({curr_vol:,.0f}) 🔴")

        # Variazione percentuale oraria
        if len(close) >= 2:
            pct_change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100.0
            extra_data['daily_var_pct'] = pct_change

        # CALCOLO SCORE GREZZO
        raw_score = (
            (st_score * 0.20) +
            (macd_score * 0.20) +
            (ha_score * 0.20) +
            (rsi_score * 0.15) +
            (ema_ma_score * 0.15) +
            (vol_score * 0.10)
        )

        # 🛑 APPLICAZIONE KILL-SWITCH ANTI-LATERALITÀ
        if is_lateral:
            # Se ADX < 20, applica penalizzazione del 50% e fissa tetto massimo a 0.45
            final_score = min(0.45, raw_score * 0.50)
        else:
            final_score = raw_score

        return signals, round(max(0.0, min(1.0, final_score)), 3), extra_data, df

    except Exception as e:
        print(f"❌ Errore durante l'analisi ETF di {ticker}: {e}")
        return signals, 0.5, extra_data, None


def create_unified_etf_report(results: List[Tuple[str, List[str], float, Dict, Optional[pd.DataFrame]]], descriptions: Dict) -> str:
    """Crea UN UNICO messaggio Telegram con tutti gli ETF analizzati"""
    if not results:
        return "📊 *AGENTE ETF LEVA - REPORT ORARIO*\nNessun ETF disponibile."
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    now_str = datetime.now().strftime('%H:%M')
    lines = [f"📊 *AGENTE ETF LEVA - MONITORAGGIO ({now_str})*\n"]
    
    for ticker, _, score, extra_data, _ in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        var_pct = extra_data.get('daily_var_pct', 0.0)
        sign = "+" if var_pct > 0 else ""
        
        url = f"https://antoniotonti.github.io/agente_borsa/flash/{ticker}.html"
        line = f"{bullet} [{ticker}]({url}) - {desc} {sign}{var_pct:.2f}% (score: {score:.3f})"
        lines.append(line)
        
    return "\n".join(lines)


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
        print("📊 AGENTE ETF A LEVA - ANALISI ORARIA (ANTI-LATERALITÀ)")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        etf_tickers, descriptions = load_etf_from_csv("titoli.csv")
        etf_results = []
        
        for ticker in etf_tickers:
            desc = descriptions.get(ticker, ticker)
            print(f"-> Analisi ETF Leva: {ticker} ({desc})")
            signals, score, extra_data, df = analyze_etf_leveraged(ticker)
            etf_results.append((ticker, signals, score, extra_data, df))
            
            if df is not None and not df.empty:
                generate_web_page(ticker, desc, "flash", df, score, signals)
            time.sleep(0.3)
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if token and chat_id and etf_results:
            print("\n📩 Invio unico messaggio report ETF su Telegram...")
            unified_report = create_unified_etf_report(etf_results, descriptions)
            send_telegram_message(token, chat_id, unified_report)
        elif not etf_results:
            print("\n⚠️ Nessun titolo ETF individuato nel file di configurazione CSV.")
                
        print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")
        
    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agente ETF - Analisi Oraria per ETF a Leva / Direzionali
Ottimizzato per strumenti a Leva (es. FTSE MIB 2x, Short/Bull):
- Timeframe: Orario (1h)
- Indicatori: Supertrend, ADX (filtro trend), MACD Veloce (8,17,9), RSI (9), Heikin Ashi, EMA/SMA.
- Report Telegram aggregato + Generazione Pagina Web con Grafico Score aggiornato.
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
    """Analisi specifica per ETF a leva su timeframe orario."""
    signals = []
    extra_data = {'daily_var_pct': 0.0}
    
    # Inizializzazione Score
    st_score = 0.5        # 15%
    macd_score = 0.5      # 15%
    ha_score = 0.5        # 20%
    rsi_score = 0.5       # 10%
    ema_ma_score = 0.5   # 15%
    trend_score = 0.5    # 10%
    adx_score = 0.5      # 10%
    vol_score = 0.5      # 5%

    try:
        tk = yf.Ticker(ticker)
        # Timeframe Orario per massima reattività
        df = tk.history(period="1mo", interval="1h", auto_adjust=True)
        
        if df.empty or len(df) < 30:
            # Fallback su daily se i dati orari non sono disponibili
            df = tk.history(period="6mo", interval="1d", auto_adjust=True)
            
        if df.empty or len(df) < 20:
            print(f"⚠️ {ticker}: Dati insufficienti ({len(df)} righe).")
            return signals, 0.5, extra_data, None

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        close = df['Close']
        volume = df['Volume']

        # 1. SUPERTREND ORARIO (15%)
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

        # 2. MACD VELOCE (8, 17, 9) (15%)
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

        # 3. HEIKIN ASHI CANDLE ANALYSIS (20%)
        ha = calculate_heikin_ashi(df)
        if len(ha) >= 10:
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

        # 4. RSI VELOCE (9) (10%)
        rsi = ta.momentum.rsi(close, window=9).dropna()
        if not rsi.empty:
            rsi_val = float(rsi.iloc[-1])
            if rsi_val > 70:
                rsi_score = 0.30  # Ipercomprato a breve
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Ipercomprato (Attenzione) ⚠️")
            elif rsi_val < 30:
                rsi_score = 0.80  # Potenziale irrobustimento
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Ipervenduto (Rimbalzo possibile) 🟢")
            elif rsi_val >= 50:
                rsi_score = 0.85
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Zona di Forza 🟢")
            else:
                rsi_score = 0.20
                signals.append(f"🟣 RSI (9): {rsi_val:.1f} - Zona di Debolezza 🔴")

        # 5. EMA10 vs MA31 (15%)
        if len(close) >= 31:
            ema10 = ta.trend.ema_indicator(close, window=10).iloc[-1]
            ma31 = ta.trend.sma_indicator(close, window=31).iloc[-1]
            if ema10 > ma31:
                ema_ma_score = 0.85
                signals.append(f"📊 Medie: EMA10 ({ema10:{fmt}}) sopra MA31 ({ma31:{fmt}}) 🟢")
            else:
                ema_ma_score = 0.15
                signals.append(f"📊 Medie: MA31 ({ma31:{fmt}}) sopra EMA10 ({ema10:{fmt}}) 🔴")

        # 6. STIMA TREND 7 PERIODI (10%)
        if len(close) >= 7:
            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
            extra_data.update({'var_percent': var_percent, 'target_price': target_price, 'stop_loss': stop_loss})
            signals.append(format_trend_line(var_percent, target_price, stop_loss))
            if var_percent > 2.0: trend_score = 1.0
            elif var_percent > 0: trend_score = 0.70
            elif var_percent > -2.0: trend_score = 0.30
            else: trend_score = 0.0

        # 7. FILTRO ADX - FORZA DEL TREND ANTI-LATERALITÀ (10%)
        if len(df) >= 20:
            adx_df = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            adx_val = float(adx_df.adx().dropna().iloc[-1])
            if adx_val >= 25:
                adx_score = 1.0
                signals.append(f"⚡ ADX: {adx_val:.1f} - Trend Forte in Corso 🟢")
            elif adx_val >= 20:
                adx_score = 0.70
                signals.append(f"⚡ ADX: {adx_val:.1f} - Trend Moderato 🟢")
            else:
                adx_score = 0.20
                signals.append(f"⚡ ADX: {adx_val:.1f} - Fase Laterale (Pericolo Leva!) ⚠️")

        # 8. VOLUMI ULTIMA CANDELA (5%)
        if len(volume) >= 20:
            avg_vol = float(volume.tail(20).mean())
            curr_vol = float(volume.iloc[-1])
            if curr_vol >= avg_vol:
                vol_score = 0.85
                signals.append(f"📊 Volumi: Sopra la media oraria ({curr_vol:,.0f}) 🟢")
            else:
                vol_score = 0.35
                signals.append(f"📊 Volumi: Sotto la media oraria ({curr_vol:,.0f}) 🔴")

        # Calcolo Variazione Percentuale Ultima Candela
        if len(close) >= 2:
            pct_change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100.0
            extra_data['daily_var_pct'] = pct_change

        # SCORE FINALE PESATO PER ETF A LEVA (100%)
        final_score = (
            (st_score * 0.15) +
            (macd_score * 0.15) +
            (ha_score * 0.20) +
            (rsi_score * 0.10) +
            (ema_ma_score * 0.15) +
            (trend_score * 0.10) +
            (adx_score * 0.10) +
            (vol_score * 0.05)
        )
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
        print("📊 AGENTE ETF A LEVA - ANALISI ORARIA")
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

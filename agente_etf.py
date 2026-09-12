#!/usr/bin/env python3
"""
Agente ETF - Analisi Multi-Timeframe (1H + 1D) per ETF a Leva / Direzionali
- Timeframe 1H: Operativo Intraday
- Timeframe 1D: Filtro di Trend di Fondo
- Pesi Dinamici e Gestione del Rischio Integrati
- Telegram: Report unico orario sintetico
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

# Nuove importazioni per l'architettura dinamica
try:
    import state_manager
    from risk_manager import apply_risk_limits, calculate_dynamic_weights
except ImportError:
    print("⚠️ Moduli state_manager o risk_manager non trovati. Assicurati che siano nella directory.")
    sys.exit(1)


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
                
        print(f"✅ CSV caricato: trovati {len(etf_tickers)} titoli ETF ({', '.join(etf_tickers)})")
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


def compute_timeframe_score(ticker: str, df: pd.DataFrame, timeframe_label: str) -> Tuple[float, List[str]]:
    signals = []
    if df is None or len(df) < 15:
        return 0.5, ["⚠️ Dati insufficienti per l'analisi."]

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    close = df['Close']
    volume = df['Volume']

    # Inizializzazione punteggi base
    scores = {
        'SUPERTREND': 0.5, 'MACD': 0.5, 'HA': 0.5,
        'RSI': 0.5, 'EMA_MA': 0.5, 'VOL': 0.5
    }

    # 1. FILTRO ADX (KILL-SWITCH ANTI-LATERALITÀ)
    adx_val = 0.0
    is_lateral = True
    if len(df) >= 15:
        adx_df = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        adx_series = adx_df.adx().dropna()
        if not adx_series.empty:
            adx_val = float(adx_series.iloc[-1])
            is_lateral = adx_val < 20.0

    if is_lateral:
        signals.append(f"⛔ ADX ({timeframe_label}): {adx_val:.1f} - FASE LATERALE (Blocco segnale) ⚠️")
    else:
        signals.append(f"⚡ ADX ({timeframe_label}): {adx_val:.1f} - Trend in corso confermato 🟢")

    # 2. SUPERTREND
    st_line, st_dir = calculate_supertrend(df, period=10, multiplier=3.0)
    last_st_dir = st_dir.iloc[-1]
    last_st_val = st_line.iloc[-1]
    fmt = ".4f" if close.iloc[-1] < 1.0 else ".2f"
    
    if last_st_dir == 1:
        scores['SUPERTREND'] = 1.0
        signals.append(f"🟢 Supertrend ({timeframe_label}): RIALZISTA (Supp: {last_st_val:{fmt}})")
    else:
        scores['SUPERTREND'] = 0.0
        signals.append(f"🔴 Supertrend ({timeframe_label}): RIBASSISTA (Res: {last_st_val:{fmt}})")

    # 3. MACD VELOCE (8, 17, 9)
    macd_obj = ta.trend.MACD(close=close, window_slow=17, window_fast=8, window_sign=9)
    m_line, s_line = macd_obj.macd().dropna(), macd_obj.macd_signal().dropna()
    if len(m_line) > 1 and len(s_line) > 1:
        m_now, s_now = float(m_line.iloc[-1]), float(s_line.iloc[-1])
        m_prev, s_prev = float(m_line.iloc[-2]), float(s_line.iloc[-2])
        
        if m_now > s_now and m_prev <= s_prev:
            scores['MACD'] = 1.0
            signals.append(f"📈 MACD ({timeframe_label}): CROSSOVER RIALZISTA 🟢")
        elif m_now < s_now and m_prev >= s_prev:
            scores['MACD'] = 0.0
            signals.append(f"📉 MACD ({timeframe_label}): CROSSOVER RIBASSISTA 🔴")
        elif m_now > s_now:
            scores['MACD'] = 0.75
            signals.append(f"🟢 MACD ({timeframe_label}): Positivo")
        else:
            scores['MACD'] = 0.25
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
                scores['HA'] = 1.0
                signals.append(f"🕯️ HA ({timeframe_label}): Forte Spinta Verde 🟢")
            else:
                scores['HA'] = 0.70
                signals.append(f"🕯️ HA ({timeframe_label}): Candela Verde 🟢")
        else:
            if upper_shadow <= (ha_range * 0.03):
                scores['HA'] = 0.0
                signals.append(f"🕯️ HA ({timeframe_label}): Forte Spinta Rossa 🔴")
            else:
                scores['HA'] = 0.30
                signals.append(f"🕯️ HA ({timeframe_label}): Candela Rossa 🔴")

    # 5. RSI VELOCE (9)
    rsi = ta.momentum.rsi(close, window=9).dropna()
    if not rsi.empty:
        rsi_val = float(rsi.iloc[-1])
        if rsi_val > 70:
            scores['RSI'] = 0.30
            signals.append(f"🟣 RSI ({timeframe_label}): {rsi_val:.1f} - Ipercomprato ⚠️")
        elif rsi_val < 30:
            scores['RSI'] = 0.80
            signals.append(f"🟣 RSI ({timeframe_label}): {rsi_val:.1f} - Ipervenduto 🟢")
        elif rsi_val >= 50:
            scores['RSI'] = 0.85
            signals.append(f"🟣 RSI ({timeframe_label}): {rsi_val:.1f} - Zona di Forza 🟢")
        else:
            scores['RSI'] = 0.20
            signals.append(f"🟣 RSI ({timeframe_label}): {rsi_val:.1f} - Zona di Debolezza 🔴")

    # 6. EMA10 vs MA31
    if len(close) >= 31:
        ema10 = ta.trend.ema_indicator(close, window=10).iloc[-1]
        ma31 = ta.trend.sma_indicator(close, window=31).iloc[-1]
        diff_pct = ((ema10 - ma31) / ma31) * 100.0
        
        if diff_pct > 0.1:
            scores['EMA_MA'] = 0.85
            signals.append(f"📊 Medie ({timeframe_label}): EMA10 > MA31 (+{diff_pct:.2f}%) 🟢")
        elif diff_pct < -0.1:
            scores['EMA_MA'] = 0.15
            signals.append(f"📊 Medie ({timeframe_label}): MA31 > EMA10 ({diff_pct:.2f}%) 🔴")
        else:
            scores['EMA_MA'] = 0.50
            signals.append(f"📊 Medie ({timeframe_label}): EMA10 e MA31 Sovrapposte ⚪")

    # 7. VOLUMI ULTIMA CANDELA
    if len(volume) >= 10:
        avg_vol = float(volume.tail(10).mean())
        curr_vol = float(volume.iloc[-1])
        if curr_vol >= avg_vol:
            scores['VOL'] = 0.85
            signals.append(f"📊 Volumi ({timeframe_label}): Sopra la media 🟢")
        else:
            scores['VOL'] = 0.35
            signals.append(f"📊 Volumi ({timeframe_label}): Sotto la media 🔴")

    # GESTIONE PESI DINAMICI
    os.makedirs('data_state', exist_ok=True)
    state_file = f"data_state/state_{ticker}_{timeframe_label}.json"
    current_state = state_manager.load_state(state_file)
    
    base_weights = {
        'SUPERTREND': 0.20, 'MACD': 0.20, 'HA': 0.20,
        'RSI': 0.15, 'EMA_MA': 0.15, 'VOL': 0.10
    }
    
    dynamic_weights = calculate_dynamic_weights(current_state, base_weights)

    # Calcolo score ponderato
    raw_score = sum(scores[ind] * dynamic_weights[ind] for ind in scores)
    
    final_score = min(0.45, raw_score * 0.50) if is_lateral else raw_score
    return round(max(0.0, min(1.0, final_score)), 3), signals


def analyze_etf_multi_timeframe(ticker: str) -> Dict:
    result = {
        'ticker': ticker,
        'score_1h': 0.5,
        'score_1d': 0.5,
        'signals_1h': [],
        'signals_1d': [],
        'daily_var_pct': 0.0,
        'df_1h': None
    }

    try:
        tk = yf.Ticker(ticker)

        df_1h = tk.history(period="12d", interval="1h", auto_adjust=True)
        if df_1h.empty or len(df_1h) < 15:
            df_1h = tk.history(period="1mo", interval="1d", auto_adjust=True)
        
        result['df_1h'] = df_1h
        df_1d = tk.history(period="6mo", interval="1d", auto_adjust=True)

        # Calcolo Score con Pesi Dinamici
        raw_score_1h, signals_1h = compute_timeframe_score(ticker, df_1h, "1H")
        raw_score_1d, signals_1d = compute_timeframe_score(ticker, df_1d, "1D")

        # RISK MANAGEMENT: Applicazione limiti di esposizione (Cap 25% suggerito per coperture)
        state_file = f"data_state/state_{ticker}_1H.json"
        current_state = state_manager.load_state(state_file)
        
        final_score_1h, risk_warning = apply_risk_limits(
            ticker=ticker,
            score=raw_score_1h,
            current_exposure=current_state.get('exposure', 0),
            max_exposure_limit=0.25 
        )
        
        if risk_warning:
            signals_1h.append(f"🛡️ RISK ALERT: {risk_warning}")

        result['score_1h'] = final_score_1h
        result['score_1d'] = raw_score_1d  # Il filtro giornaliero rimane puro, senza cap
        result['signals_1h'] = signals_1h
        result['signals_1d'] = signals_1d

        # Salvataggio Stato per ricalibrazione futura
        current_state['last_score'] = final_score_1h
        current_state['exposure'] = current_state.get('exposure', 0) + (final_score_1h * 0.1) # Simulazione accumulo
        state_manager.save_state(state_file, current_state)

        # Calcolo Variazione Percentuale
        try:
            fast_info = getattr(tk, 'fast_info', {})
            last_price = fast_info.get('lastPrice', None)
            prev_close = fast_info.get('previousClose', None)

            if last_price is None or np.isnan(last_price):
                if df_1h is not None and not df_1h.empty:
                    last_price = float(df_1h['Close'].iloc[-1])

            if prev_close is not None and not np.isnan(prev_close) and prev_close > 0:
                pct_change = ((last_price - prev_close) / prev_close) * 100.0
            elif df_1d is not None and len(df_1d) >= 2:
                p_close = float(df_1d['Close'].iloc[-2])
                p_last = float(df_1d['Close'].iloc[-1])
                pct_change = ((p_last - p_close) / p_close) * 100.0
            else:
                pct_change = 0.0

            result['daily_var_pct'] = pct_change
        except Exception as e_var:
            print(f"⚠️ Errore calcolo variazione per {ticker}: {e_var}")
            result['daily_var_pct'] = 0.0

    except Exception as e:
        print(f"❌ Errore durante l'analisi ETF di {ticker}: {e}")

    return result


def create_unified_etf_report(results: List[Dict], descriptions: Dict) -> str:
    if not results:
        return "📊 *AGENTE ETF LEVA - REPORT ORARIO*\nNessun ETF disponibile."
    
    sorted_results = sorted(results, key=lambda x: x['score_1h'], reverse=True)
    now_str = datetime.now().strftime('%H:%M')
    lines = [f"📊 *AGENTE ETF LEVA - MONITORAGGIO ({now_str})*\n"]
    
    for res in sorted_results:
        ticker = res['ticker']
        desc = descriptions.get(ticker, ticker)
        s_1h = res['score_1h']
        s_1d = res['score_1d']
        b_1h = get_bullet(s_1h)
        b_1d = get_bullet(s_1d)
        
        var_pct = res['daily_var_pct']
        sign = "+" if var_pct > 0 else ""
        
        url = f"https://antoniotonti.github.io/agente_borsa/flash/{ticker}.html"
        
        line = (
            f"🔹 [{ticker}]({url}) - *{desc}* (Oggi: {sign}{var_pct:.2f}%)\n"
            f"   ├ ⚡ *1H Intraday:* {b_1h} Score: `{s_1h:.3f}`\n"
            f"   └ 📈 *1D Daily:*    {b_1d} Score: `{s_1d:.3f}`"
        )
        
        if any("RISK ALERT" in s for s in res['signals_1h']):
            line += "\n   🛡️ *Tetto Rischio Raggiunto*"
        elif s_1h >= 0.65 and s_1d < 0.45:
            line += "\n   ⚠️ *ATTENZIONE:* Intraday rialzista ma controtrend Daily!"
        elif s_1h >= 0.70 and s_1d >= 0.70:
            line += "\n   🔥 *CONFLUENZA RIALZISTA (1H + 1D)* 🟢"
            
        lines.append(line + "\n")
        
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
        print("📊 AGENTE ETF - PESI DINAMICI & RISK MANAGEMENT")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        etf_tickers, descriptions = load_etf_from_csv("titoli.csv")
        etf_results = []
        
        for ticker in etf_tickers:
            desc = descriptions.get(ticker, ticker)
            print(f"-> Analisi ETF Leva: {ticker} ({desc})")
            
            res = analyze_etf_multi_timeframe(ticker)
            etf_results.append(res)
            
            if res['df_1h'] is not None and not res['df_1h'].empty:
                combined_signals = ["--- TIMEFRAME 1H (INTRADAY) ---"] + res['signals_1h'] + \
                                   ["", "--- TIMEFRAME 1D (DAILY) ---"] + res['signals_1d']
                generate_web_page(ticker, desc, "flash", res['df_1h'], res['score_1h'], combined_signals)
                
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

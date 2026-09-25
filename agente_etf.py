#!/usr/bin/env python3
"""
Agente ETF - Analisi Oraria con ADX Regime

Caratteristiche:
- Solo timeframe Hourly (1H)
- 11 indicatori (analyst escluso, adx_regime al suo posto)
- Pesi ottimizzati per "velocità" (delta EMA, trend, RSI più pesanti)
- DB separato: etf_history.db
- Orizzonte verifica: 5 giorni
- Auto-tuning dei pesi attivo
- Solo ticker con tipo == "ETF" da titoli.csv
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
from config import load_titoli_csv
from analysis_utils import (
    calculate_heikin_ashi,
    get_bullet,
    calculate_trend_estimate,
    format_trend_line
)
from web_generator import generate_web_page

from db_manager import (
    set_db_path, init_db, save_previsione, get_previsioni_da_verificare,
    registra_verifica, get_pesi_correnti, direzione_da_score,
    classifica_esito,
)
from auto_tuner import evaluate_and_tune

# ============================================================================
# COSTANTI
# ============================================================================
DB_PATH = "etf_history.db"
RISK_PER_TRADE_PCT = 2.0
ORIZZONTE_VERIFICA_GIORNI = 5
ETF_MIN_POINTS = 32  # minimo candele hourly per analisi

# Pesi di default ETF (ottimizzati per velocità, senza analyst, con adx_regime)
DEFAULT_WEIGHTS_ETF = {
    "ema_ma": 0.12,
    "trend": 0.15,
    "delta_ema_ma": 0.15,
    "ha_force": 0.15,
    "ha_state": 0.10,
    "zigzag": 0.05,
    "vol": 0.05,
    "close_change": 0.05,
    "rsi": 0.05,
    "macd": 0.03,
    "adx_regime": 0.10,
}


# ============================================================================
# HELPER: ZIGZAG
# ============================================================================
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


# ============================================================================
# HELPER: RISCHIO
# ============================================================================
def compute_risk_metrics(df: pd.DataFrame, current_price: Optional[float] = None) -> Dict:
    default = {
        'atr': 0.0, 'sl': 0.0, 'tp': 0.0, 'sizing': 0.0,
        'vol_bullet': '⚪', 'vol_label': 'N/D', 'max_capital': 0.0,
    }
    if df is None or len(df) < 15:
        return default
    try:
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()
        if current_price is None:
            current_price = float(close.iloc[-1])
        atr_ind = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        atr_series = atr_ind.average_true_range().dropna()
        if atr_series.empty:
            return default
        atr = float(atr_series.iloc[-1])
        if atr <= 0 or current_price <= 0:
            return default
        sl = current_price - 1.5 * atr
        tp = current_price + 3.0 * atr
        sizing = (atr * 1.5) / current_price * 100.0
        if sizing < 2.0:
            vol_bullet, vol_label = "🟢", "Bassa"
        elif sizing <= 4.0:
            vol_bullet, vol_label = "⚪", "Media"
        else:
            vol_bullet, vol_label = "🔴", "Alta"
        max_capital = min(RISK_PER_TRADE_PCT / sizing * 100.0, 100.0) if sizing > 0 else 100.0
        return {
            'atr': round(atr, 3), 'sl': round(sl, 3), 'tp': round(tp, 3),
            'sizing': round(sizing, 2), 'vol_bullet': vol_bullet,
            'vol_label': vol_label, 'max_capital': round(max_capital, 0),
        }
    except Exception as e:
        print(f"   ⚠️ Errore calcolo rischio: {e}")
        return default


# ============================================================================
# HELPER: ADX REGIME
# ============================================================================
def compute_adx_regime(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Calcola ADX e restituisce (score, messaggio).
    Score:
      ADX < 20   → 0.20 (laterale, penalizza)
      ADX 20-25  → 0.50 (neutro)
      ADX 25-40  → 1.00 (trend forte, favorisce)
      ADX > 40   → 0.70 (trend esaurito, cautela)
    """
    if df is None or len(df) < 15:
        return 0.5, "ADX: N/D"

    try:
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        close = df['Close'].squeeze()

        adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        adx_series = adx_obj.adx().dropna()
        if adx_series.empty:
            return 0.5, "ADX: N/D"

        adx_val = float(adx_series.iloc[-1])

        if adx_val < 20:
            return 0.20, f"ADX {adx_val:.1f} - Laterale 🔴"
        elif adx_val < 25:
            return 0.50, f"ADX {adx_val:.1f} - Neutro ⚪"
        elif adx_val <= 40:
            return 1.00, f"ADX {adx_val:.1f} - Trend Forte 🟢"
        else:
            return 0.70, f"ADX {adx_val:.1f} - Trend Esaurito ⚠️"
    except Exception as e:
        print(f"   ⚠️ Errore ADX: {e}")
        return 0.5, "ADX: N/D"


# ============================================================================
# MOTORE DI ANALISI (ETF — senza analyst, con adx_regime)
# ============================================================================
def analyze_df_engine(
    df: pd.DataFrame,
    pesi: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], float, Dict, Dict]:
    signals = []
    extra_data = {}

    if pesi is None:
        pesi = dict(DEFAULT_WEIGHTS_ETF)

    ema_ma_score = trend_score = ema_ma_delta_score = 0.5
    ha_force_score = ha_state_score = zigzag_score = vol_score = 0.5
    close_change_score = rsi_score = macd_score = 0.5
    adx_regime_score = 0.5
    analyst_score = 0.5  # non usato nei pesi ETF, ma presente per coerenza sub_scores

    close = df['Close']
    volume = df['Volume']

    # 1. EMA10 vs MA31
    clean_ema, clean_ma = None, None
    if len(close) >= 31:
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

    # 2. STIMA TREND
    if len(close) >= 10:
        var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=7)
        extra_data.update({'var_percent': var_percent, 'target_price': target_price, 'stop_loss': stop_loss})
        signals.append(format_trend_line(var_percent, target_price, stop_loss))
        if var_percent > 3.0: trend_score = 1.0
        elif var_percent > 0.0: trend_score = 0.75
        elif var_percent == 0.0: trend_score = 0.50
        elif var_percent > -3.0: trend_score = 0.25
        else: trend_score = 0.0

    # 3. DELTA % EMA10/MA31
    if clean_ema is not None and clean_ma is not None and len(clean_ma) >= 20:
        common_idx = clean_ema.index.intersection(clean_ma.index)
        delta_series = ((clean_ema.loc[common_idx] - clean_ma.loc[common_idx]) / clean_ma.loc[common_idx]) * 100.0
        curr_delta = float(delta_series.iloc[-1])
        lookback_len = min(63, len(delta_series))
        avg_delta = float(delta_series.tail(lookback_len).abs().mean())
        sign = "+" if curr_delta > 0 else ""
        signals.append(f"📐 Delta EMA10/MA31: {sign}{curr_delta:.2f}% (Media Abs: {avg_delta:.2f}%)")
        if curr_delta > 0:
            ema_ma_delta_score = 1.0 if (avg_delta > 0 and curr_delta >= avg_delta * 1.5) else (0.80 if curr_delta >= avg_delta else 0.60)
        else:
            abs_curr = abs(curr_delta)
            ema_ma_delta_score = 0.0 if (avg_delta > 0 and abs_curr >= avg_delta * 1.5) else (0.20 if abs_curr >= avg_delta else 0.40)

    # 4 & 5. HEIKIN ASHI
    ha = calculate_heikin_ashi(df)
    if len(ha) >= 20:
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
        lookback_ha = min(63, len(ha_bodies))
        avg_body = float(ha_bodies.tail(lookback_ha).mean())
        ratio_body = (ha_bodies.iloc[-1] / avg_body) if avg_body > 0 else 1.0
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
        signals.append(f"🕯️ Heikin Ashi: {ha_desc} - Forza Corpo: {ratio_body:.2f}x media")

    # 6. ZIGZAG
    zz_trend = calculate_zigzag_trend(df, deviation_pct=5.0)
    zigzag_score = 1.0 if zz_trend == 1 else (0.0 if zz_trend == -1 else 0.5)
    zz_desc = "Rialzista 🟢" if zz_trend == 1 else ("Ribassista 🔴" if zz_trend == -1 else "Neutro ⚪")
    signals.append(f"⚡ ZigZag (5%): Trend {zz_desc}")

    # 7. VOLUME
    if len(volume) >= 20:
        lookback_vol = min(63, len(volume))
        avg_vol = float(volume.tail(lookback_vol).mean())
        curr_vol = float(volume.iloc[-1])
        if curr_vol > avg_vol * 1.5:
            vol_score = 1.0
            vol_desc = "Volumi in forte aumento (>150% media) 🟢"
        elif curr_vol >= avg_vol:
            vol_score = 0.75
            vol_desc = "Volumi sopra la media 🟢"
        else:
            vol_score = 0.35
            vol_desc = "Volumi sotto la media 🔴"
        signals.append(f"📊 Volumi: {curr_vol:,.0f} vs Media {avg_vol:,.0f} ({vol_desc})")

    # 8. RSI
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

    # 9. MACD
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

    # 10. ADX REGIME
    adx_regime_score, adx_msg = compute_adx_regime(df)
    signals.append(f"💪 {adx_msg}")

    sub_scores = {
        "ema_ma": ema_ma_score,
        "trend": trend_score,
        "delta_ema_ma": ema_ma_delta_score,
        "ha_force": ha_force_score,
        "ha_state": ha_state_score,
        "zigzag": zigzag_score,
        "vol": vol_score,
        "close_change": close_change_score,
        "rsi": rsi_score,
        "macd": macd_score,
        "adx_regime": adx_regime_score,
        "analyst": analyst_score,  # peso 0 nei default ETF, ma lo salviamo per coerenza
    }

    final_score = sum(sub_scores[k] * pesi.get(k, 0.0) for k in sub_scores)
    final_score = max(0.0, min(1.0, final_score))

    return signals, round(final_score, 3), extra_data, sub_scores


# ============================================================================
# ANALISI TICKER ETF (solo Hourly)
# ============================================================================
def analyze_etf_ticker(
    ticker: str,
    pesi: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], float, Dict, Optional[pd.DataFrame]]:
    extra_data = {'hourly_var_pct': 0.0, 'risk_hourly': {}}
    try:
        tk = yf.Ticker(ticker)

        df_h = tk.history(period="1mo", interval="1h", auto_adjust=True)
        if df_h.empty or len(df_h) < ETF_MIN_POINTS:
            print(f"⚠️ {ticker}: dati hourly insufficienti.")
            return [], 0.5, extra_data, None

        if isinstance(df_h.columns, pd.MultiIndex):
            df_h.columns = df_h.columns.get_level_values(0)

        df_h = df_h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        last_price = float(df_h['Close'].iloc[-1])
        prev_close = float(df_h['Close'].iloc[-2]) if len(df_h) >= 2 else last_price

        pct_change = ((last_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        extra_data['hourly_var_pct'] = pct_change
        extra_data['last_price'] = last_price
        extra_data['risk_hourly'] = compute_risk_metrics(df_h, current_price=last_price)

        signals, score, extra_d, sub_scores = analyze_df_engine(df_h, pesi=pesi)
        extra_data.update(extra_d)
        extra_data['sub_scores'] = sub_scores

        return signals, score, extra_data, df_h

    except Exception as e:
        print(f"❌ Errore analisi {ticker}: {e}")
        return [], 0.5, extra_data, None


# ============================================================================
# VERIFICA PREVISIONI SCADUTE
# ============================================================================
def verifica_previsioni_scadute() -> None:
    previsioni = get_previsioni_da_verificare()
    if not previsioni:
        print("   ✅ Nessuna previsione da verificare")
        return
    print(f"   🔍 {len(previsioni)} previsioni da verificare...")
    for prev in previsioni:
        ticker = prev['ticker']
        prezzo_in = prev['prezzo_emissione']
        direzione = prev['direzione']
        try:
            tk = yf.Ticker(ticker)
            fast_info = getattr(tk, 'fast_info', {})
            prezzo_ora = fast_info.get('lastPrice', None)
            if prezzo_ora is None or (isinstance(prezzo_ora, float) and np.isnan(prezzo_ora)):
                df_tmp = tk.history(period="5d", interval="1h", auto_adjust=True)
                if df_tmp.empty:
                    continue
                prezzo_ora = float(df_tmp['Close'].iloc[-1])
            if prezzo_in <= 0:
                continue
            rendimento_pct = ((prezzo_ora - prezzo_in) / prezzo_in) * 100.0
            esito = classifica_esito(rendimento_pct)
            esito_corretto = (esito == direzione)
            registra_verifica(prev['id'], prezzo_ora, rendimento_pct, esito, esito_corretto)
            icona = "✅" if esito_corretto else "❌"
            print(f"      {icona} {ticker}: prev={direzione}, real={esito} ({rendimento_pct:+.2f}%)")
        except Exception as e:
            print(f"      ❌ Errore verifica {ticker}: {e}")


# ============================================================================
# FORMATTAZIONE REPORT
# ============================================================================
def _format_risk_block(risk: Dict) -> List[str]:
    if not risk or risk.get('atr', 0.0) == 0.0:
        return ["   ├ 🎯 SL: N/D | TP: N/D", "   └ ⚪ Volatilità: N/D | Size: N/D"]
    return [
        f"   ├ 🎯 SL: {risk['sl']:.3f} | TP: {risk['tp']:.3f}",
        f"   ├ {risk['vol_bullet']} Volatilità: {risk['vol_label']} (ATR {risk['atr']:.3f}) | Size: {risk['sizing']:.2f}%",
        f"   └ 💰 Rischio {RISK_PER_TRADE_PCT:.0f}% → max *{risk['max_capital']:.0f}%* del capitale",
    ]


def _build_single_ticker_block(
    ticker: str, score: float, extra_data: Dict, descriptions: Dict
) -> str:
    desc = descriptions.get(ticker, ticker)
    bullet = get_bullet(score)
    var_pct = extra_data.get('hourly_var_pct', 0.0)
    sign = "+" if var_pct > 0 else ""
    url = f"https://antoniotonti.github.io/agente_borsa/etf/{ticker}.html"
    risk_h = extra_data.get('risk_hourly', {})

    parts = []
    parts.append(f"🔹 [{ticker}]({url}) - *{desc}* (Oggi: {sign}{var_pct:.2f}%)")
    parts.append("")
    parts.append(f"   ⚡ *1H:* {bullet} `{score:.3f}`")
    parts.extend(_format_risk_block(risk_h))
    parts.append("")
    return "\n".join(parts)


def create_report_blocks(
    title: str,
    results: List[Tuple[str, List[str], float, Dict, Optional[pd.DataFrame]]],
    descriptions: Dict
) -> Tuple[str, List[str]]:
    if not results:
        return title, ["Nessun dato disponibile."]
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    blocks = []
    for ticker, _, score, extra_data, _ in sorted_results:
        blocks.append(_build_single_ticker_block(ticker, score, extra_data, descriptions))
    return title, blocks


def create_etf_report(results, descriptions):
    now_str = datetime.now().strftime("%H:%M")
    return create_report_blocks(f"📊 *ETF ORARIO ({now_str})*", results, descriptions)


# ============================================================================
# TELEGRAM — chunking
# ============================================================================
def _chunk_blocks(title: str, blocks: List[str], max_len: int = 3800) -> List[str]:
    chunks = []
    current = title + "\n"
    current_len = len(current)
    for block in blocks:
        block_len = len(block) + 1
        if block_len > max_len:
            if current_len > len(title) + 1:
                chunks.append(current.rstrip())
                current = title + "\n"
                current_len = len(current)
            chunks.append(block)
            continue
        if current_len + block_len > max_len:
            chunks.append(current.rstrip())
            current = title + "\n"
            current_len = len(current)
        current += block + "\n"
        current_len += block_len
    if current_len > len(title) + 1:
        chunks.append(current.rstrip())
    return chunks if chunks else [title]


def send_telegram_message(token: str, chat_id: str, payload_data) -> bool:
    if isinstance(payload_data, tuple):
        title, blocks = payload_data
        chunks = _chunk_blocks(title, blocks)
    else:
        chunks = [payload_data]
    total = len(chunks)
    success = True
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i, chunk in enumerate(chunks, start=1):
        if total > 1:
            text = f"{chunk}\n\n_({i}/{total})_"
        else:
            text = chunk
        payload = {
            "chat_id": chat_id, "text": text,
            "parse_mode": "Markdown", "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                print(f"❌ Errore Telegram ({resp.status_code}) chunk {i}/{total}: {resp.text}")
                success = False
            else:
                print(f"   ✅ Chunk {i}/{total} inviato ({len(text)} caratteri)")
        except Exception as e:
            print(f"❌ Errore Telegram chunk {i}/{total}: {e}")
            success = False
        time.sleep(1.5)
    return success


# ============================================================================
# PROCESSO ETF
# ============================================================================
def process_etf_group(tickers: List[str], descriptions: Dict) -> List:
    results = []
    if not tickers:
        return results

    categoria = "ETF"
    pesi = get_pesi_correnti(categoria, DEFAULT_WEIGHTS_ETF)
    print(f"\n📊 ANALISI ETF ({len(tickers)} strumenti)")
    print(f"   Pesi attivi:")
    for ind, p in sorted(pesi.items(), key=lambda x: -x[1]):
        print(f"      {ind:15s}: {p:.4f}")

    for ticker in tickers:
        signals, score, extra_data, df_h = analyze_etf_ticker(ticker, pesi=pesi)
        results.append((ticker, signals, score, extra_data, df_h))

        if df_h is not None and not df_h.empty:
            desc = descriptions.get(ticker, ticker)
            generate_web_page(ticker, desc, "etf", df_h, score, signals)
            prezzo_emissione = extra_data.get('last_price', float(df_h['Close'].iloc[-1]))
            sub_scores = extra_data.get('sub_scores', {})
            try:
                save_previsione(
                    ticker=ticker, categoria=categoria,
                    prezzo_emissione=prezzo_emissione, score=score,
                    direzione=direzione_da_score(score),
                    sub_scores=sub_scores,
                    orizzonte_giorni=ORIZZONTE_VERIFICA_GIORNI,
                )
                print(f"      💾 {ticker}: salvato (score={score:.3f})")
            except Exception as e:
                print(f"      ⚠️ {ticker}: errore DB: {e}")
        time.sleep(0.5)
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    start_time = time.time()
    try:
        print("=" * 60)
        print("📊 AGENTE ETF - ANALISI ORARIA con ADX Regime")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)

        set_db_path(DB_PATH)
        init_db()

        print("\n🔍 VERIFICA PREVISIONI SCADUTE")
        verifica_previsioni_scadute()

        print("\n🎯 AUTO-TUNING PESI")
        evaluate_and_tune("ETF")

        # Carica titoli e filtra solo ETF
        _, _, descriptions, etf_list = load_titoli_csv()

        etf_results = process_etf_group(etf_list, descriptions)

        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if token and chat_id and etf_results:
            print("\n📩 Invio report ETF...")
            send_telegram_message(token, chat_id, create_etf_report(etf_results, descriptions))

        print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")

    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")


if __name__ == "__main__":
    main()

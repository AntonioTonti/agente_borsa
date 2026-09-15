#!/usr/bin/env python3
"""
Agente di Trading - Analisi Giornaliera (FLASH)

Format Telegram: Ticker in evidenza con 2 blocchi (1D Daily + 1H Intraday).
Ogni blocco mostra score, SL/TP, volatilità e size rischio massima.

Integrazione DB + Auto-tuning dei pesi (learning rate 0.02).
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

from db_manager import (
    init_db,
    save_previsione,
    get_previsioni_da_verificare,
    registra_verifica,
    get_pesi_correnti,
    direzione_da_score,
    classifica_esito,
    DEFAULT_WEIGHTS,
)
from auto_tuner import evaluate_and_tune

# ============================================================================
# COSTANTI
# ============================================================================
RISK_PER_TRADE_PCT = 2.0
ORIZZONTE_VERIFICA_GIORNI = 3


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
# HELPER: RATING ANALISTI
# ============================================================================
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

        atr_ind = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        )
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
            'atr': round(atr, 3),
            'sl': round(sl, 3),
            'tp': round(tp, 3),
            'sizing': round(sizing, 2),
            'vol_bullet': vol_bullet,
            'vol_label': vol_label,
            'max_capital': round(max_capital, 0),
        }
    except Exception as e:
        print(f"   ⚠️ Errore calcolo rischio: {e}")
        return default


# ============================================================================
# MOTORE DI ANALISI (pesi dinamici)
# ============================================================================
def analyze_df_engine(
    df: pd.DataFrame,
    tk: Optional[yf.Ticker] = None,
    pesi: Optional[Dict[str, float]] = None
) -> Tuple[List[str], float, Dict, Dict]:
    """
    Motore universale di calcolo score e indicatori.
    Ritorna: (signals, score, extra_data, sub_scores)
    """
    signals = []
    extra_data = {}

    if pesi is None:
        pesi = dict(DEFAULT_WEIGHTS)

    ema_ma_score = trend_score = analyst_score = ema_ma_delta_score = 0.5
    ha_force_score = ha_state_score = zigzag_score = vol_score = 0.5
    close_change_score = rsi_score = macd_score = 0.5

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

    # 3. RATING ANALISTI
    if tk is not None:
        analyst_score, analyst_msg = get_analyst_rating_score(tk)
        signals.append(f"🎯 {analyst_msg}")

    # 4. DELTA % EMA10/MA31
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

    # 5 & 6. HEIKIN ASHI
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

    # 7. ZIGZAG
    zz_trend = calculate_zigzag_trend(df, deviation_pct=5.0)
    zigzag_score = 1.0 if zz_trend == 1 else (0.0 if zz_trend == -1 else 0.5)
    zz_desc = "Rialzista 🟢" if zz_trend == 1 else ("Ribassista 🔴" if zz_trend == -1 else "Neutro ⚪")
    signals.append(f"⚡ ZigZag (5%): Trend {zz_desc}")

    # 8. VOLUME
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

    # 9. RSI
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

    # 10. MACD
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

    # Sub-scores
    sub_scores = {
        "ema_ma": ema_ma_score,
        "trend": trend_score,
        "analyst": analyst_score,
        "delta_ema_ma": ema_ma_delta_score,
        "ha_force": ha_force_score,
        "ha_state": ha_state_score,
        "zigzag": zigzag_score,
        "vol": vol_score,
        "close_change": close_change_score,
        "rsi": rsi_score,
        "macd": macd_score,
    }

    # Score finale dinamico
    final_score = sum(sub_scores[k] * pesi.get(k, 0.0) for k in sub_scores)
    final_score = max(0.0, min(1.0, final_score))

    return signals, round(final_score, 3), extra_data, sub_scores


# ============================================================================
# ANALISI TICKER
# ============================================================================
def analyze_flash_ticker(
    ticker: str,
    pesi: Optional[Dict[str, float]] = None
) -> Tuple[List[str], float, float, Dict, Optional[pd.DataFrame]]:
    extra_data = {'daily_var_pct': 0.0, 'risk_daily': {}, 'risk_hourly': {}}
    try:
        tk = yf.Ticker(ticker)

        # --- DATI DAILY ---
        df_d = tk.history(period="6mo", interval="1d", auto_adjust=True)
        if df_d.empty or len(df_d) < DAILY_MIN_POINTS:
            print(f"⚠️ {ticker}: Dati daily vuoti o insufficienti.")
            return [], 0.5, 0.5, extra_data, None

        df_d = df_d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        fast_info = getattr(tk, 'fast_info', {})
        last_price = fast_info.get('lastPrice', None)
        prev_close = fast_info.get('previousClose', None)

        if last_price is None or np.isnan(last_price):
            last_price = float(df_d['Close'].iloc[-1])

        today_date = datetime.now().date()
        last_df_date = df_d.index[-1].date()

        if prev_close is None or np.isnan(prev_close) or prev_close <= 0:
            if last_df_date == today_date and len(df_d) >= 2:
                prev_close = float(df_d['Close'].iloc[-2])
            elif last_df_date < today_date and len(df_d) >= 1:
                prev_close = float(df_d['Close'].iloc[-1])
            else:
                prev_close = last_price

        if last_df_date == today_date:
            df_d.iloc[-1, df_d.columns.get_loc('Close')] = last_price

        pct_change = ((last_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        extra_data['daily_var_pct'] = pct_change
        extra_data['last_price'] = last_price

        extra_data['risk_daily'] = compute_risk_metrics(df_d, current_price=last_price)

        signals_d, score_d, extra_d, sub_scores_d = analyze_df_engine(df_d, tk=tk, pesi=pesi)
        extra_data.update(extra_d)
        extra_data['sub_scores'] = sub_scores_d

        # --- DATI HOURLY (manteniamo score_h, verrà rimosso in Step 5) ---
        score_h = 0.5
        df_h = tk.history(period="1mo", interval="1h", auto_adjust=True)
        if not df_h.empty and len(df_h) >= 20:
            df_h = df_h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            extra_data['risk_hourly'] = compute_risk_metrics(df_h)
            _, score_h, _, _ = analyze_df_engine(df_h, tk=None, pesi=pesi)

        return signals_d, score_d, score_h, extra_data, df_d

    except Exception as e:
        print(f"❌ Errore durante l'analisi di {ticker}: {e}")
        return [], 0.5, 0.5, extra_data, None


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
                df_tmp = tk.history(period="5d", interval="1d", auto_adjust=True)
                if df_tmp.empty:
                    print(f"      ⚠️ {ticker}: prezzo non disponibile, skip")
                    continue
                prezzo_ora = float(df_tmp['Close'].iloc[-1])

            if prezzo_in <= 0:
                print(f"      ⚠️ {ticker}: prezzo emissione non valido, skip")
                continue

            rendimento_pct = ((prezzo_ora - prezzo_in) / prezzo_in) * 100.0
            esito = classifica_esito(rendimento_pct)
            esito_corretto = (esito == direzione)

            registra_verifica(
                previsione_id=prev['id'],
                prezzo_verifica=prezzo_ora,
                rendimento_pct=rendimento_pct,
                esito=esito,
                esito_corretto=esito_corretto,
            )

            icona = "✅" if esito_corretto else "❌"
            print(f"      {icona} {ticker}: prev={direzione}, real={esito} ({rendimento_pct:+.2f}%)")

        except Exception as e:
            print(f"      ❌ Errore verifica {ticker}: {e}")


# ============================================================================
# FORMATTAZIONE REPORT
# ============================================================================
def _format_risk_block(risk: Dict) -> List[str]:
    if not risk or risk.get('atr', 0.0) == 0.0:
        return [
            f"   ├ 🎯 SL: N/D | TP: N/D",
            f"   └ ⚪ Volatilità: N/D | Size: N/D",
        ]

    return [
        f"   ├ 🎯 SL: {risk['sl']:.3f} | TP: {risk['tp']:.3f}",
        f"   ├ {risk['vol_bullet']} Volatilità: {risk['vol_label']} (ATR {risk['atr']:.3f}) | Size: {risk['sizing']:.2f}%",
        f"   └ 💰 Rischio {RISK_PER_TRADE_PCT:.0f}% → max *{risk['max_capital']:.0f}%* del capitale",
    ]


def create_daily_report_section(
    title: str,
    results: List[Tuple[str, List[str], float, float, Dict, Optional[pd.DataFrame]]],
    descriptions: Dict
) -> str:
    if not results:
        return f"{title}\nNessun dato disponibile."

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = [f"{title}\n"]

    for ticker, _, score_d, score_h, extra_data, _ in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet_d = get_bullet(score_d)
        bullet_h = get_bullet(score_h)
        var_pct = extra_data.get('daily_var_pct', 0.0)
        sign = "+" if var_pct > 0 else ""

        url = f"https://antoniotonti.github.io/agente_borsa/flash/{ticker}.html"

        risk_d = extra_data.get('risk_daily', {})
        risk_h = extra_data.get('risk_hourly', {})

        lines.append(f"🔹 [{ticker}]({url}) - *{desc}* (Oggi: {sign}{var_pct:.2f}%)")
        lines.append("")

        lines.append(f"   📈 *1D Daily:* {bullet_d} `{score_d:.3f}`")
        lines.extend(_format_risk_block(risk_d))
        lines.append("")

        lines.append(f"   ⚡ *1H Intraday:* {bullet_h} `{score_h:.3f}`")
        lines.extend(_format_risk_block(risk_h))
        lines.append("")

    return "\n".join(lines)


def create_portfolio_daily_report(results, descriptions) -> str:
    now_str = datetime.now().strftime("%H:%M")
    return create_daily_report_section(f"💰 *PORTAFOGLIO GIORNALIERO ({now_str})*", results, descriptions)


def create_watchlist_daily_report(results, descriptions) -> str:
    now_str = datetime.now().strftime("%H:%M")
    return create_daily_report_section(f"👁️ *OSSERVATI GIORNALIERI ({now_str})*", results, descriptions)


def create_etf_daily_report(results, descriptions) -> str:
    now_str = datetime.now().strftime("%H:%M")
    return create_daily_report_section(f"📊 *ETF ({now_str})*", results, descriptions)


# ============================================================================
# TELEGRAM
# ============================================================================
def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    MAX_LENGTH = 3800
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
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                print(f"❌ Errore Telegram ({resp.status_code}): {resp.text}")
                success = False
        except Exception as e:
            print(f"❌ Errore invio Telegram: {e}")
            success = False
        time.sleep(1)

    return success


# ============================================================================
# GRUPPO TICKER
# ============================================================================
def process_ticker_group(tickers: List[str], categoria: str, descriptions: Dict) -> List:
    """Analizza un gruppo di ticker, genera pagine web, salva su DB."""
    results = []
    if not tickers:
        return results

    pesi = get_pesi_correnti(categoria, DEFAULT_WEIGHTS)
    print(f"\n📊 ANALISI {categoria} ({len(tickers)} ticker)")
    print(f"   Pesi attivi:")
    for ind, p in sorted(pesi.items(), key=lambda x: -x[1]):
        print(f"      {ind:15s}: {p:.4f}")

    for ticker in tickers:
        signals_d, score_d, score_h, extra_data, df_d = analyze_flash_ticker(ticker, pesi=pesi)
        results.append((ticker, signals_d, score_d, score_h, extra_data, df_d))

        if df_d is not None and not df_d.empty:
            desc = descriptions.get(ticker, ticker)
            generate_web_page(ticker, desc, "flash", df_d, score_d, signals_d)

            prezzo_emissione = extra_data.get('last_price', float(df_d['Close'].iloc[-1]))
            sub_scores = extra_data.get('sub_scores', {})

            try:
                save_previsione(
                    ticker=ticker,
                    categoria=categoria,
                    prezzo_emissione=prezzo_emissione,
                    score=score_d,
                    direzione=direzione_da_score(score_d),
                    sub_scores=sub_scores,
                    orizzonte_giorni=ORIZZONTE_VERIFICA_GIORNI,
                )
                print(f"      💾 {ticker}: salvato (score={score_d:.3f})")
            except Exception as e:
                print(f"      ⚠️ {ticker}: errore salvataggio DB: {e}")
        time.sleep(0.5)

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    start_time = time.time()
    try:
        print("=" * 60)
        print("📊 AGENTE DI TRADING - ANALISI FLASH")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)

        # Init DB
        init_db()

        # Verifica previsioni scadute
        print("\n🔍 VERIFICA PREVISIONI SCADUTE")
        verifica_previsioni_scadute()

        # Auto-tuning pesi
        print("\n🎯 AUTO-TUNING PESI")
        for cat in ["PORTAFOGLIO", "WATCHLIST", "ETF"]:
            evaluate_and_tune(cat)

        # Carica titoli
        portfolio, watchlist, descriptions, etf_list = load_titoli_csv()

        # Analizza i 3 gruppi
        portfolio_results = process_ticker_group(portfolio, "PORTAFOGLIO", descriptions)
        watchlist_results = process_ticker_group(watchlist, "WATCHLIST", descriptions)
        etf_results = process_ticker_group(etf_list, "ETF", descriptions)

        # Invio Telegram
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
                time.sleep(2)
            if etf_results:
                print("\n📩 Invio report ETF...")
                send_telegram_message(token, chat_id, create_etf_daily_report(etf_results, descriptions))

        print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")

    except Exception as e:
        print(f"❌ ERRORE GENERALE: {e}")


if __name__ == "__main__":
    main()

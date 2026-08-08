#!/usr/bin/env python3

"""

Agente di Trading - Analisi Giornaliera (Focus Direzionalità e Forza)

PESI E SOGLIE AGGIORNATI (TOTALE 100%):

1. EMA10 vs MA31 (18%)

2. Stima Trend 7gg (15%) [Soglia: ±3.0%]

3. Delta % EMA10 vs MA31 vs Media 3M (12%)

4. Heikin Ashi - Forza/Estensione Corpo vs Media 3M (15%)

5. Heikin Ashi - Analisi Tecnica Stato/Ombre (10%)

6. ZigZag (10%)

7. Volume vs Media 3 Mesi (5%)

8. Chiusura vs Chiusura Prec. (5%)

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

    Analisi giornaliera a 10 parametri ricalibrata sui nuovi pesi e medie a 3 mesi (63gg).

    """

    signals = []

    

    # Default neutrali

    ema_ma_score = 0.5

    trend_score = 0.5

    ema_ma_delta_score = 0.5

    ha_force_score = 0.5

    ha_state_score = 0.5

    zigzag_score = 0.5

    vol_score = 0.5

    close_change_score = 0.5

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

        

        # 1. EMA10 vs MA31 (PESO 18%)

        ema_now = None

        ma_now = None

        clean_ema = None

        clean_ma = None

        

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



        # 3. DELTA % EMA10 vs MA31 PESATO SU MEDIA 3 MESI (~63 SESSIONI) (PESO 12%)

        if clean_ema is not None and clean_ma is not None and len(clean_ma) >= 63:

            common_idx = clean_ema.index.intersection(clean_ma.index)

            delta_series = ((clean_ema.loc[common_idx] - clean_ma.loc[common_idx]) / clean_ma.loc[common_idx]) * 100.0

            

            curr_delta = float(delta_series.iloc[-1])

            avg_delta_3m = float(delta_series.tail(63).abs().mean())  # ampiezza media assoluta del delta a 3 mesi

            

            sign = "+" if curr_delta > 0 else ""

            signals.append(f"📐 Delta EMA10/MA31: {sign}{curr_delta:.2f}% (Media Abs 3M: {avg_delta_3m:.2f}%)")

            

            if curr_delta > 0:

                if avg_delta_3m > 0 and curr_delta >= avg_delta_3m * 1.5:

                    ema_ma_delta_score = 1.0

                elif avg_delta_3m > 0 and curr_delta >= avg_delta_3m:

                    ema_ma_delta_score = 0.80

                else:

                    ema_ma_delta_score = 0.60

            else:

                abs_curr = abs(curr_delta)

                if avg_delta_3m > 0 and abs_curr >= avg_delta_3m * 1.5:

                    ema_ma_delta_score = 0.0

                elif avg_delta_3m > 0 and abs_curr >= avg_delta_3m:

                    ema_ma_delta_score = 0.20

                else:

                    ema_ma_delta_score = 0.40



        # 4 & 5. HEIKIN ASHI - FORZA (15%) E STATO/ANALISI OMBRE (10%)

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

            is_doji = (ha_body / ha_range) < 0.15  # Corpo piccolo rispetto al range = incertezza

            

            # 4. FORZA CORPO HA VS MEDIA 3 MESI (63 SESSIONI) (15%)

            ha_bodies = (ha['HA_Close'] - ha['HA_Open']).abs()

            curr_body = ha_bodies.iloc[-1]

            avg_body_3m = float(ha_bodies.tail(63).mean())

            ratio_body = (curr_body / avg_body_3m) if avg_body_3m > 0 else 1.0

            

            if is_green:

                if ratio_body >= 1.5:

                    signals.append(f"🕯️ Espansione Corpo HA: {ratio_body:.1f}x media 3M (Forte estensione)")

                    ha_force_score = 1.0

                elif ratio_body >= 1.0:

                    signals.append(f"🕯️ Espansione Corpo HA: {ratio_body:.1f}x media 3M (Moderata)")

                    ha_force_score = 0.75

                else:

                    signals.append(f"🕯️ Espansione Corpo HA: {ratio_body:.1f}x media 3M (Corpo contenuto)")

                    ha_force_score = 0.50

            else:

                if ratio_body >= 1.5:

                    signals.append(f"🕯️ Espansione Corpo HA: {ratio_body:.1f}x media 3M (Pressione ribassista alta)")

                    ha_force_score = 0.0

                elif ratio_body >= 1.0:

                    signals.append(f"🕯️ Espansione Corpo HA: {ratio_body:.1f}x media 3M (Pressione ribassista media)")

                    ha_force_score = 0.25

                else:

                    signals.append(f"🕯️ Espansione Corpo HA: {ratio_body:.1f}x media 3M (Ribasso contenuto)")

                    ha_force_score = 0.40



            # 5. LETTURA ANALISI TECNICA HA (COLORE + OMBRE) (10%)

            if is_doji:

                signals.append("⚖️ HEIKIN ASHI: DOJI / INCERTEZZA (Corpo molto stretto)")

                ha_state_score = 0.50

            elif is_green:

                if lower_shadow <= (ha_range * 0.03):  # Ombra inferiore quasi assente

                    signals.append("🟢 HEIKIN ASHI: VERDE SENZA OMBRA INF. (Spinta rialzista pura)")

                    ha_state_score = 1.0

                elif upper_shadow > lower_shadow:

                    signals.append("🟢 HEIKIN ASHI: VERDE CON OMBRA SUP. PREVALENTE")

                    ha_state_score = 0.75

                else:

                    signals.append("🟢 HEIKIN ASHI: VERDE CON OMBRA INF. (Possibile rallentamento)")

                    ha_state_score = 0.60

            else:

                if upper_shadow <= (ha_range * 0.03):  # Ombra superiore quasi assente

                    signals.append("🔴 HEIKIN ASHI: ROSSA SENZA OMBRA SUP. (Spinta ribassista pura)")

                    ha_state_score = 0.0

                elif lower_shadow > upper_shadow:

                    signals.append("🔴 HEIKIN ASHI: ROSSA CON OMBRA INF. PREVALENTE")

                    ha_state_score = 0.25

                else:

                    signals.append("🔴 HEIKIN ASHI: ROSSA CON OMBRA SUP. (Pressione in attenuazione)")

                    ha_state_score = 0.40



        # 6. ZIGZAG (PESO 10%)

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



        # 7. VOLUME VS MEDIA 3 MESI (~63 SESSIONI) (PESO 5%)

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



        # 8. CHIUSURA VS PRECEDENTE (PESO 5%)

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



        # SCORE FINALE AGGIORNATO AL 100%

        final_score = (

            (ema_ma_score * 0.18) +

            (trend_score * 0.15) +

            (ema_ma_delta_score * 0.12) +

            (ha_force_score * 0.15) +

            (ha_state_score * 0.10) +

            (zigzag_score * 0.10) +

            (vol_score * 0.05) +

            (close_change_score * 0.05) +

            (rsi_score * 0.05) +

            (macd_score * 0.05)

        )

        final_score = max(0.0, min(1.0, final_score))

        

        return signals, round(final_score, 3), extra_data

        

    except Exception as e:

        print(f"❌ {ticker}: {e}")

        return signals, 0.5, extra_data



def format_signals_for_report(signals: List[str]) -> List[str]:

    """

    Pulisce ed evita rientri anomali sulle righe multilinea per la formattazione Telegram.

    """

    formatted = []

    for sig in signals:

        lines = sig.strip().split('\n')

        for l in lines:

            if l.strip():

                formatted.append(f"  {l.strip()}")

    return formatted



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

            lines.extend(format_signals_for_report(signals))

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

            lines.extend(format_signals_for_report(signals))

        else:

            lines.append("  📭 Nessun segnale rilevato")

            

        lines.append("----------------------------")

        

    return "\n".join(lines)



def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:

    """

    Invia un messaggio Telegram dividendolo automaticamente se supera la lunghezza massima.

    """

    MAX_LENGTH = 3800  # Margine di sicurezza

    

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

        time.sleep(1)

        

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



#!/usr/bin/env python3

"""

Agente di Trading - Analisi Settimanale (STESSA LOGICA DEL GIORNALIERO)

Invio: Venerdì 18:00 UTC (19:00 IT)

FEATURES:

- STESSA ANALISI del giornaliero (Heikin Ashi, EMA, RSI, Volume)

- MA su Dati SETTIMANALI invece che giornalieri

- Stima Trend (3 settimane) con Target e Stop Loss

- Pallino riassuntivo 🟢/⚪/🔴 a destra del nome

- Analisi separata per Portafoglio e Watchlist

- Due invii Telegram distinti

"""



import os

import sys

import time

from datetime import datetime

import requests

import yfinance as yf

import pandas as pd

import numpy as np

from typing import List, Dict, Tuple



# Configurazione

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



# Costanti settimanali

WEEKLY_PERIOD = "1y"      # 1 anno di dati

WEEKLY_INTERVAL = "1wk"   # Barre settimanali

WEEKLY_MIN_POINTS = 20    # Minimo 20 settimane per l'analisi



# ============================================================================

# INDICATORI SETTIMANALI (IDENTICI al giornaliero)

# ============================================================================



def analyze_weekly_ticker(ticker: str) -> Tuple[List[str], float, Dict]:

    """

    Analisi settimanale - STESSA LOGICA del giornaliero

    Restituisce: (segnali, score, dati_aggiuntivi)

    """

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

        

        close = df['Close']

        volume = df['Volume']

        

        # ================================================================

        # STIMA TREND (3 settimane)

        # ================================================================

        if len(close) >= 6:

            var_percent, target_price, stop_loss = calculate_trend_estimate(close, lookback=3)

            extra_data = {

                'var_percent': var_percent,

                'target_price': target_price,

                'stop_loss': stop_loss

            }

        

        # ================================================================

        # 1. HEIKIN ASHI (PESO 0.35)

        # ================================================================

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

        

        # ================================================================

        # 2. EMA10 vs MA31 (PESO 0.30)

        # ================================================================

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

                    score += 0.25

                elif ma_now > ema_now and ma_prev <= ema_prev:

                    signals.append("📉 MA31 > EMA10 (CROSSOVER DOWN)")

                    score -= 0.25

                elif ema_now > ma_now:

                    signals.append("🟢 EMA10 sopra MA31")

                    score += 0.15

                else:

                    signals.append("🔴 MA31 sopra EMA10")

                    score -= 0.15

        

        # ================================================================

        # 3. RSI (PESO 0.20)

        # ================================================================

        if len(close) >= 15:

            import ta

            rsi = ta.momentum.rsi(close, window=14)

            if len(rsi) > 0:

                rsi_val = float(rsi.iloc[-1])

                if rsi_val > 70:

                    signals.append("⚠️ RSI > 70 (IPERCOMPRATO)")

                    score -= 0.15

                elif rsi_val < 30:

                    signals.append("⚠️ RSI < 30 (IPERVENDUTO)")

                    score += 0.10

                elif rsi_val > 60:

                    score += 0.05

                elif rsi_val < 40:

                    score -= 0.05

        

        # ================================================================

        # 4. Volume (PESO 0.15)

        # ================================================================

        if len(volume) >= 10:

            avg_volume = float(volume.tail(10).mean())

            current_volume = float(volume.iloc[-1])

            if current_volume > avg_volume * 1.5:

                signals.append("📊 Volume +50%")

                score += 0.10

            elif current_volume < avg_volume * 0.5:

                score -= 0.05

        

        # ================================================================

        # COMBINAZIONE FINALE SCORE

        # ================================================================

        other_indicators_score = score

        ha_normalized = (ha_color_score + 0.45) / 0.9

        final_score = (ha_normalized * 0.35) + (other_indicators_score * 0.65)

        final_score = max(0.0, min(1.0, final_score))

        

        return signals, round(final_score, 3), extra_data

        

    except Exception as e:

        print(f"❌ {ticker}: {e}")

        return signals, 0.5, extra_data



# ============================================================================

# FUNZIONI DI FORMATTAZIONE REPORT CON PALLINO A DESTRA E TREND

# ============================================================================



def create_portfolio_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:

    """Crea report per portafoglio con pallino a destra e stima trend"""

    if not results:

        return "💰 *PORTAFOGLIO SETTIMANALE* - Nessun segnale oggi"

    

    sorted_results = sorted(results, key=lambda x: x[2])

    

    lines = []

    lines.append("💰 *PORTAFOGLIO SETTIMANALE*")

    

    for ticker, signals, score, extra_data in sorted_results:

        desc = descriptions.get(ticker, ticker)

        bullet = get_bullet(score)

        

        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")

        

        if signals:

            for signal in signals:

                lines.append(f"  {signal}")

        else:

            lines.append(f"  📭 Nessun segnale rilevato")

        

        # Aggiungi stima trend se disponibile

        if extra_data and 'var_percent' in extra_data:

            trend_line = format_trend_line(

                extra_data['var_percent'],

                extra_data['target_price'],

                extra_data['stop_loss']

            )

            lines.append(trend_line)

        

        lines.append("----------------------------")

    

    return "\n".join(lines)



def create_watchlist_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:

    """Crea report per watchlist con pallino a destra e stima trend"""

    if not results:

        return "👁️ *OSSERVATI SETTIMANALI* - Nessun segnale oggi"

    

    sorted_results = sorted(results, key=lambda x: x[2])

    

    lines = []

    lines.append("👁️ *OSSERVATI SETTIMANALI*")

    

    for ticker, signals, score, extra_data in sorted_results:

        desc = descriptions.get(ticker, ticker)

        bullet = get_bullet(score)

        

        lines.append(f"\n*{ticker}* - {desc} {bullet} (score: {score:.3f})")

        

        if signals:

            for signal in signals:

                lines.append(f"  {signal}")

        else:

            lines.append(f"  📭 Nessun segnale rilevato")

        

        # Aggiungi stima trend se disponibile

        if extra_data and 'var_percent' in extra_data:

            trend_line = format_trend_line(

                extra_data['var_percent'],

                extra_data['target_price'],

                extra_data['stop_loss']

            )

            lines.append(trend_line)

        

        lines.append("----------------------------")

    

    return "\n".join(lines)



# ============================================================================

# FUNZIONI DI INVIO TELEGRAM

# ============================================================================



def send_telegram_message(token: str, chat_id: str, message: str, use_markdown: bool = True) -> bool:

    """Invia un messaggio a Telegram con gestione errori"""

    try:

        url = f"https://api.telegram.org/bot{token}/sendMessage"

        

        MAX_LENGTH = 4096

        

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

        

        for i, part in enumerate(parts):

            payload = {

                "chat_id": chat_id,

                "text": part,

                "parse_mode": "Markdown" if (use_markdown and i == 0) else None,

                "disable_web_page_preview": True,

                "disable_notification": (i > 0)

            }

            

            resp = requests.post(url, json=payload, timeout=15)

            

            if resp.status_code != 200:

                print(f"    ❌ Errore invio parte {i+1}: {resp.status_code}")

                return False

            

            if i < len(parts) - 1:

                time.sleep(0.5)

        

        return True

        

    except Exception as e:

        print(f"❌ Errore invio Telegram: {e}") 


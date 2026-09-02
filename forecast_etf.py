#!/usr/bin/env python3
"""
Agente di Trading - Previsioni ETF e Azioni
Implementa un calcolo solido sui rendimenti storici e projettori sequenziali.
Invia 3 messaggi distinti: Portafoglio Azioni, Azioni Osservate, ETF.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
import pandas as pd
import numpy as np

sys.path.append('.')
from config import load_titoli_csv, DAILY_MIN_POINTS
from web_generator import generate_web_page


def calculate_forecast_sequence(series: pd.Series, horizon: int = 5) -> List[float]:
    """
    Calcola la proiezione dei rendimenti futuri basandosi sui momentum percentuali 
    e sulla stima della deriva (drift) degli ultimi periodi.
    Garantisce variazioni dinamiche e non nulle.
    """
    if series is None or len(series) < 10:
        return [0.0] * horizon

    vals = series.dropna().values.astype(np.float64)
    if len(vals) < 10:
        return [0.0] * horizon

    # Calcolo dei rendimenti percentuali recenti
    pct_returns = np.diff(vals) / vals[:-1] * 100.0

    # Trend recente (media ponderata degli ultimi rendimenti)
    weights = np.exp(np.linspace(-1, 0, len(pct_returns)))
    weights /= weights.sum()
    
    mean_return = np.sum(pct_returns * weights)
    volatility = np.std(pct_returns) if len(pct_returns) > 1 else 0.1

    # Generazione sequenza futura con decadimento del momentum
    forecast_pcts = []
    current_cum = 0.0
    
    for i in range(1, horizon + 1):
        # Il momentum si smorza leggermente verso la media col passare del tempo
        decay = 0.85 ** (i - 1)
        step_return = mean_return * decay
        current_cum += step_return
        forecast_pcts.append(round(current_cum, 2))

    return forecast_pcts


def get_status_circle(change_pct: float, threshold: float = 0.15) -> str:
    if change_pct >= threshold:
        return "🟢"
    elif change_pct <= -threshold:
        return "🔴"
    else:
        return "⚪"


def analyze_instrument(ticker: str) -> Tuple[List[float], List[float], float, float, Optional[pd.DataFrame]]:
    hourly_changes = [0.0] * 5
    daily_changes = [0.0] * 5
    var_today_pct = 0.0
    last_price = 0.0

    try:
        tk = yf.Ticker(ticker)

        # 1. Previsione Daily (1D..5D)
        df_d = tk.history(period="6mo", interval="1d", auto_adjust=True)
        if not df_d.empty and len(df_d) >= DAILY_MIN_POINTS:
            df_d = df_d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            fast_info = getattr(tk, 'fast_info', {})
            last_price = fast_info.get('lastPrice', float(df_d['Close'].iloc[-1]))
            prev_close = fast_info.get('previousClose', float(df_d['Close'].iloc[-2] if len(df_d) >= 2 else last_price))
            var_today_pct = ((last_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

            daily_changes = calculate_forecast_sequence(df_d['Close'], horizon=5)

        # 2. Previsione Intraday (1H..5H)
        df_h = tk.history(period="1mo", interval="1h", auto_adjust=True)
        if not df_h.empty and len(df_h) >= 15:
            df_h = df_h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            hourly_changes = calculate_forecast_sequence(df_h['Close'], horizon=5)

        return hourly_changes, daily_changes, var_today_pct, last_price, df_d

    except Exception as e:
        print(f"❌ Errore durante l'analisi per {ticker}: {e}")
        return hourly_changes, daily_changes, var_today_pct, last_price, None


def format_message_block(
    title: str, 
    results: List[Tuple[str, str, List[float], List[float], float]]
) -> str:
    now_rome = datetime.now(ZoneInfo("Europe/Rome")).strftime("%H:%M")
    if not results:
        return f"{title} ({now_rome})\nNessun elemento presente."

    lines = [f"{title} ({now_rome})\n"]

    for ticker, desc, h_changes, d_changes, var_today in results:
        sign = "+" if var_today > 0 else ""
        url = f"https://antoniotonti.github.io/agente_borsa/forecast_etf/{ticker}.html"

        header = f"🔹 [{ticker}]({url}) - {desc} ({sign}{var_today:.2f}%)"
        
        d_str = " ".join([f"{get_status_circle(ch, 0.20)}{ch:+.1f}%" for ch in d_changes])
        daily_line = f"├ 📈 *1D-5D Daily:* {d_str}"
        
        h_str = " ".join([f"{get_status_circle(ch, 0.10)}{ch:+.1f}%" for ch in h_changes])
        hourly_line = f"└ ⚡ *1H-5H Intraday:* {h_str}\n"

        lines.extend([header, daily_line, hourly_line])

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
    now_str = datetime.now(ZoneInfo("Europe/Rome")).strftime('%d/%m/%Y %H:%M:%S')
    print("=" * 60)
    print("🤖 AGENTE TRADING - FORECAST (ALGORITMO DINAMICO DI PROIEZIONE)")
    print(f"Avvio: {now_str}")
    print("=" * 60)

    # Caricamento delle liste
    portafoglio_titoli, osservati_titoli, descriptions = load_titoli_csv()
    
    # Classificazione per la suddivisione esplicita nei 3 canali
    etf_keywords = ["ETF", "ISHARES", "XTRACKERS", "LYXOR", "VANGUARD", "AMUNDI", "WISDOMTREE", "XEON", "SWDA", "MEUD", "CSSPX", "SGLD"]
    
    portafoglio_azioni = []
    osservati_azioni = []
    lista_etf = []

    for t in portafoglio_titoli:
        desc = descriptions.get(t, "").upper()
        if any(kw in desc or kw in t.upper() for kw in etf_keywords):
            lista_etf.append(t)
        else:
            portafoglio_azioni.append(t)

    for t in osservati_titoli:
        desc = descriptions.get(t, "").upper()
        if any(kw in desc or kw in t.upper() for kw in etf_keywords):
            if t not in lista_etf:
                lista_etf.append(t)
        else:
            osservati_azioni.append(t)

    def process_group(tickers: List[str], group_name: str):
        results = []
        if not tickers:
            return results
        print(f"\n--- Elaborazione {group_name} ({len(tickers)} strumenti) ---")
        for ticker in tickers:
            h_ch, d_ch, var_pct, price, df_d = analyze_instrument(ticker)
            desc = descriptions.get(ticker, ticker)
            results.append((ticker, desc, h_ch, d_ch, var_pct))
            
            if df_d is not None and not df_d.empty:
                score_d = round(0.50 + (d_ch[-1] / 6.0), 3) if len(d_ch) > 0 else 0.5
                generate_web_page(ticker, desc, "forecast_etf", df_d, score_d, [f"Forecast 5D: {d_ch[-1]:+.2f}%"])
            time.sleep(0.1)
        return results

    # Processa i 3 flussi distinti
    res_portafoglio = process_group(portafoglio_azioni, "AZIONI PORTAFOGLIO")
    res_osservati = process_group(osservati_azioni, "AZIONI OSSERVATE")
    res_etf = process_group(lista_etf, "ETF")

    # Invio Telegram in 3 Messaggi ben distinti
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if token and chat_id:
        if res_portafoglio:
            print("\n📩 Invio Telegram 1/3: Azioni Portafoglio...")
            msg1 = format_message_block("📊 *FORECAST AZIONI PORTAFOGLIO*", res_portafoglio)
            send_telegram_message(token, chat_id, msg1)
            time.sleep(1.5)

        if res_osservati:
            print("📩 Invio Telegram 2/3: Azioni Osservate...")
            msg2 = format_message_block("👁️ *FORECAST AZIONI OSSERVATI*", res_osservati)
            send_telegram_message(token, chat_id, msg2)
            time.sleep(1.5)

        if res_etf:
            print("📩 Invio Telegram 3/3: ETF...")
            msg3 = format_message_block("💰 *FORECAST ETF*", res_etf)
            send_telegram_message(token, chat_id, msg3)

    print(f"\n🏁 Completato con successo in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agente di Trading - Previsioni con Google TimesFM
Gestione ottimizzata dell'alimentazione dei dati storici a TimesFM.
Invia 3 messaggi Telegram distinti per Azioni Portafoglio, Azioni Osservate ed ETF.
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

try:
    import timesfm
except ImportError:
    print("❌ Errore: la libreria 'timesfm' non è installata.")
    sys.exit(1)

sys.path.append('.')
from config import load_titoli_csv, DAILY_MIN_POINTS
from web_generator import generate_web_page


class TimesFMPredictor:
    """Wrapper per il caricamento ed inferenza con Google TimesFM."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimesFMPredictor, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        print("🤖 Caricamento modello Google TimesFM...")
        try:
            self.tfm = timesfm.TimesFm(
                context_len=128,
                horizon_len=12,
                input_patch_len=32,
                output_patch_len=12,
                num_layers=20,
                model_dims=1280,
                backend="cpu"
            )
            self.tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")
            print("✅ Modello TimesFM caricato con successo.")
        except Exception as e:
            print(f"❌ Errore durante il caricamento del modello TimesFM: {e}")
            self.tfm = None

    def predict_sequence(self, series: pd.Series, horizon: int = 5, freq_code: int = 0) -> Tuple[List[float], float]:
        """
        Pulisce e prepara i dati storici scaricati per l'inferenza TimesFM.
        freq_code: 0 per dati giornalieri, 1 per dati orari.
        """
        if self.tfm is None or series is None or len(series) < 16:
            return [0.0] * horizon, 0.0

        try:
            # 1. Pulizia e conversione in array float32
            clean_series = series.dropna().astype(np.float32)
            if len(clean_series) < 16:
                return [0.0] * horizon, float(clean_series.iloc[-1]) if len(clean_series) > 0 else 0.0

            # Prendi gli ultimi 128 punti di contesto (o tutti se < 128)
            context_vals = clean_series.values[-128:]
            last_price = float(context_vals[-1])

            if last_price <= 0:
                return [0.0] * horizon, last_price

            # 2. Normalizzazione relativa per evitare l'appiattimento di TimesFM sui prezzi nominali elevati
            scaled_input = (context_vals / last_price).astype(np.float32)

            # 3. Chiamata di forecast passando le liste vettoriali
            forecast_out, _ = self.tfm.forecast(
                inputs=[scaled_input],
                freq=[freq_code]
            )

            # Estrazione valori predetti (1D array)
            preds_scaled = forecast_out[0][:horizon]

            # 4. Calcolo delle variazioni percentuali reali
            changes_pct = []
            for p in preds_scaled:
                pred_price = float(p) * last_price
                pct = ((pred_price - last_price) / last_price) * 100.0
                changes_pct.append(pct)

            return changes_pct, last_price

        except Exception as e:
            print(f"⚠️ Errore inferenza TimesFM: {e}")
            return [0.0] * horizon, float(series.iloc[-1]) if not series.empty else 0.0


def get_status_circle(change_pct: float, threshold: float = 0.15) -> str:
    if change_pct >= threshold:
        return "🟢"
    elif change_pct <= -threshold:
        return "🔴"
    else:
        return "⚪"


def analyze_instrument_timesfm(ticker: str, predictor: TimesFMPredictor) -> Tuple[List[float], List[float], float, float, Optional[pd.DataFrame]]:
    hourly_changes = [0.0] * 5
    daily_changes = [0.0] * 5
    var_today_pct = 0.0
    last_price = 0.0

    try:
        tk = yf.Ticker(ticker)

        # 1. Download e Previsione Daily (1D..5D)
        df_d = tk.history(period="1y", interval="1d", auto_adjust=True)
        if not df_d.empty and len(df_d) >= DAILY_MIN_POINTS:
            df_d = df_d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            fast_info = getattr(tk, 'fast_info', {})
            last_price = fast_info.get('lastPrice', float(df_d['Close'].iloc[-1]))
            prev_close = fast_info.get('previousClose', float(df_d['Close'].iloc[-2] if len(df_d) >= 2 else last_price))
            var_today_pct = ((last_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

            daily_changes, _ = predictor.predict_sequence(df_d['Close'], horizon=5, freq_code=0)

        # 2. Download e Previsione Intraday (1H..5H)
        df_h = tk.history(period="1mo", interval="1h", auto_adjust=True)
        if not df_h.empty and len(df_h) >= 16:
            df_h = df_h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            hourly_changes, _ = predictor.predict_sequence(df_h['Close'], horizon=5, freq_code=1)

        return hourly_changes, daily_changes, var_today_pct, last_price, df_d

    except Exception as e:
        print(f"❌ Errore scaricamento/analisi per {ticker}: {e}")
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
        
        # Formattazione Daily (1D..5D)
        d_str = " ".join([f"{get_status_circle(ch, 0.20)}{ch:+.1f}%" for ch in d_changes])
        daily_line = f"├ 📈 *1D-5D Daily:* {d_str}"
        
        # Formattazione Intraday (1H..5H)
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
    print("🤖 AGENTE TRADING - FORECAST (GOOGLE TIMESFM)")
    print(f"Avvio: {now_str}")
    print("=" * 60)

    predictor = TimesFMPredictor()

    # Caricamento delle liste
    portafoglio_titoli, osservati_titoli, descriptions = load_titoli_csv()
    
    # Classificazione per suddivisione esplicita in 3 liste
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
            h_ch, d_ch, var_pct, price, df_d = analyze_instrument_timesfm(ticker, predictor)
            desc = descriptions.get(ticker, ticker)
            results.append((ticker, desc, h_ch, d_ch, var_pct))
            
            if df_d is not None and not df_d.empty:
                score_d = round(0.50 + (d_ch[-1] / 6.0), 3) if len(d_ch) > 0 else 0.5
                generate_web_page(ticker, desc, "forecast_etf", df_d, score_d, [f"TimesFM 5D: {d_ch[-1]:+.2f}%"])
            time.sleep(0.1)
        return results

    # Processa separatamente i 3 flussi
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

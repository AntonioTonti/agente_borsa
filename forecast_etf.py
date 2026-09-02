#!/usr/bin/env python3
"""
Agente di Trading - Previsioni ETF con Google TimesFM (FORECAST_ETF)
Corretto per fuso orario italiano (Europe/Rome) e vettorizzazione TimesFM.
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
from zoneinfo import ZoneInfo

# Importazione TimesFM di Google
try:
    import timesfm
except ImportError:
    print("❌ Errore: la libreria 'timesfm' non è installata.")
    sys.exit(1)

sys.path.append('.')
from config import load_titoli_csv, DAILY_MIN_POINTS
from web_generator import generate_web_page


class TimesFMPredictor:
    """Wrapper singleton per il caricamento e l'inferenza del modello Google TimesFM."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimesFMPredictor, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        print("🤖 Caricamento modello Google TimesFM in corso...")
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

    def predict_sequence(self, series: pd.Series, horizon: int = 5, freq: int = 0) -> Tuple[List[float], float]:
        """
        Genera le previsioni per i successivi 'horizon' step temporali.
        freq: 0 per dati giornalieri, 1 per dati orari.
        """
        if self.tfm is None or len(series) < 32:
            return [0.0] * horizon, float(series.iloc[-1]) if not series.empty else 0.0

        try:
            vals = series.values.astype(np.float32)
            last_price = float(vals[-1])

            # Inizializzazione input vettoriale per TimesFM
            forecast_df = self.tfm.forecast(
                inputs=[vals],
                freq=[freq]
            )
            
            # Estrazione array delle previsioni puntuali
            if isinstance(forecast_df, tuple):
                preds = forecast_df[0][0][:horizon]
            else:
                preds = forecast_df[0][:horizon]

            changes_pct = []
            for p in preds:
                pred_val = float(p)
                pct = ((pred_val - last_price) / last_price) * 100.0
                changes_pct.append(pct)

            return changes_pct, last_price
        except Exception as e:
            print(f"⚠️ Errore durante l'inferenza TimesFM: {e}")
            return [0.0] * horizon, float(series.iloc[-1]) if not series.empty else 0.0


def get_status_circle(change_pct: float, threshold: float = 0.2) -> str:
    if change_pct >= threshold:
        return "🟢"
    elif change_pct <= -threshold:
        return "🔴"
    else:
        return "⚪"


def analyze_etf_timesfm(ticker: str, predictor: TimesFMPredictor) -> Tuple[List[float], List[float], float, float, Optional[pd.DataFrame]]:
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

            daily_changes, _ = predictor.predict_sequence(df_d['Close'], horizon=5, freq=0)

        # 2. Previsione Intraday (1H..5H)
        df_h = tk.history(period="1mo", interval="1h", auto_adjust=True)
        if not df_h.empty and len(df_h) >= 32:
            df_h = df_h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            hourly_changes, _ = predictor.predict_sequence(df_h['Close'], horizon=5, freq=1)

        return hourly_changes, daily_changes, var_today_pct, last_price, df_d

    except Exception as e:
        print(f"❌ Errore durante l'analisi TimesFM per {ticker}: {e}")
        return hourly_changes, daily_changes, var_today_pct, last_price, None


def format_etf_message_block(
    title: str, 
    results: List[Tuple[str, str, List[float], List[float], float]]
) -> str:
    # Gestione Orario con fuso italiano Europe/Rome
    now_rome = datetime.now(ZoneInfo("Europe/Rome")).strftime("%H:%M")
    if not results:
        return f"{title} ({now_rome})\nNessun elemento disponibile."

    lines = [f"{title} ({now_rome})\n"]

    for ticker, desc, h_changes, d_changes, var_today in results:
        sign = "+" if var_today > 0 else ""
        url = f"https://antoniotonti.github.io/agente_borsa/forecast_etf/{ticker}.html"

        header = f"🔹 [{ticker}]({url}) - {desc} ({sign}{var_today:.2f}%)"
        d_str = " ".join([f"{get_status_circle(ch, 0.3)}{ch:+.1f}%" for ch in d_changes])
        daily_line = f"├ 📈 *1D-5D Daily:* {d_str}"
        h_str = " ".join([f"{get_status_circle(ch, 0.15)}{ch:+.1f}%" for ch in h_changes])
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
    print("🤖 AGENTE TRADING - FORECAST ETF (GOOGLE TIMESFM)")
    print(f"Avvio: {now_str}")
    print("=" * 60)

    predictor = TimesFMPredictor()

    # Carica la lista titoli/ETF
    portfolio_titoli, watchlist_titoli, descriptions = load_titoli_csv()

    # Filtra o seleziona gli strumenti per l'analisi
    portfolio_results = []
    if portfolio_titoli:
        print("\n💰 ANALISI PORTAFOGLIO")
        for ticker in portfolio_titoli:
            h_ch, d_ch, var_pct, price, df_d = analyze_etf_timesfm(ticker, predictor)
            desc = descriptions.get(ticker, ticker)
            portfolio_results.append((ticker, desc, h_ch, d_ch, var_pct))
            
            if df_d is not None and not df_d.empty:
                score_d = round(0.50 + (d_ch[-1] / 6.0), 3) if len(d_ch) > 0 else 0.5
                generate_web_page(ticker, desc, "forecast_etf", df_d, score_d, [f"TimesFM 5D: {d_ch[-1]:+.2f}%"])
            time.sleep(0.2)

    watchlist_results = []
    if watchlist_titoli:
        print("\n👁️ ANALISI WATCHLIST")
        for ticker in watchlist_titoli:
            h_ch, d_ch, var_pct, price, df_d = analyze_etf_timesfm(ticker, predictor)
            desc = descriptions.get(ticker, ticker)
            watchlist_results.append((ticker, desc, h_ch, d_ch, var_pct))
            
            if df_d is not None and not df_d.empty:
                score_d = round(0.50 + (d_ch[-1] / 6.0), 3) if len(d_ch) > 0 else 0.5
                generate_web_page(ticker, desc, "forecast_etf", df_d, score_d, [f"TimesFM 5D: {d_ch[-1]:+.2f}%"])
            time.sleep(0.2)

    # Invio messaggi separati
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if token and chat_id:
        if portfolio_results:
            msg_port = format_etf_message_block("📊 *FORECAST ETF PORTAFOGLIO*", portfolio_results)
            print("\n📩 Invio Telegram Portafoglio...")
            send_telegram_message(token, chat_id, msg_port)
            time.sleep(1.5)

        if watchlist_results:
            msg_watch = format_etf_message_block("👁️ *FORECAST ETF OSSERVATI*", watchlist_results)
            print("\n📩 Invio Telegram Osservati...")
            send_telegram_message(token, chat_id, msg_watch)

    print(f"\n🏁 Completato in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()

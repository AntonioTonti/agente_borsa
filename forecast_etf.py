#!/usr/bin/env python3
"""
Agente di Trading - Previsioni ETF con Google TimesFM (FORECAST_ETF)
Format Telegram: Ticker in evidenza, 1D Daily (1a riga) e 1H Intraday (2a riga indentata).
Utilizza il modello TimesFM di Google per stimare il trend di prezzo futuro.
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

# Importazione TimesFM di Google
try:
    import timesfm
except ImportError:
    print("❌ Errore: la libreria 'timesfm' non è installata. Installa con: pip install timesfm")
    sys.exit(1)

sys.path.append('.')
from config import load_titoli_csv, DAILY_MIN_POINTS
from analysis_utils import get_bullet, format_trend_line
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
            # Inizializzazione del modello TimesFM 1.0 / 2.0 PyTorch
            self.tfm = timesfm.TimesFm(
                context_len=128,
                horizon_len=12,
                input_patch_len=32,
                output_patch_len=12,
                num_layers=20,
                model_dims=1280,
                backend="cpu"  # Impostare "gpu" se disponibile ambiente CUDA
            )
            # Caricamento checkpoint da HuggingFace
            self.tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")
            print("✅ Modello TimesFM caricato con successo.")
        except Exception as e:
            print(f"❌ Errore durante il caricamento del modello TimesFM: {e}")
            self.tfm = None

    def predict_change(self, series: pd.Series, horizon: int = 5) -> Tuple[float, float]:
        """
        Riceve una serie storica di prezzi di chiusura e restituisce:
        - percent_change: variazione percentuale prevista a fine orizzonte
        - predicted_price: prezzo finale stimato
        """
        if self.tfm is None or len(series) < 32:
            return 0.0, float(series.iloc[-1]) if not series.empty else 0.0

        try:
            # TimesFM richiede una lista di serie/array numerici
            input_data = [series.values.astype(np.float32)]
            forecast, _ = self.tfm.forecast(input_data, forecast_horizon=horizon)
            
            last_price = float(series.iloc[-1])
            predicted_price = float(forecast[0][-1])
            
            percent_change = ((predicted_price - last_price) / last_price) * 100.0
            return percent_change, predicted_price
        except Exception as e:
            print(f"⚠️ Errore durante l'inferenza TimesFM: {e}")
            return 0.0, float(series.iloc[-1])


def convert_forecast_to_score(change_pct: float) -> float:
    """
    Mappa la variazione % prevista da TimesFM in uno score normalizzato da 0.0 a 1.0.
    - Change >= +3.0% -> Score 1.00
    - Change == 0.0%  -> Score 0.50
    - Change <= -3.0% -> Score 0.00
    """
    if change_pct >= 3.0:
        return 1.00
    elif change_pct <= -3.0:
        return 0.00
    else:
        # Mappatura lineare tra -3.0% e +3.0% -> [0.0, 1.0]
        return round(0.50 + (change_pct / 6.0), 3)


def analyze_etf_timesfm(ticker: str, predictor: TimesFMPredictor) -> Tuple[List[str], float, float, Dict, Optional[pd.DataFrame]]:
    """
    Analizza un ETF generando previsioni TimesFM per 1D (Daily) e 1H (Intraday).
    """
    extra_data = {'daily_var_pct': 0.0}
    signals_daily = []
    
    try:
        tk = yf.Ticker(ticker)
        
        # --- 1. PREVISIONE DAILY (1D) ---
        df_d = tk.history(period="6mo", interval="1d", auto_adjust=True)
        if df_d.empty or len(df_d) < DAILY_MIN_POINTS:
            print(f"⚠️ {ticker}: Dati daily vuoti o insufficienti.")
            return [], 0.5, 0.5, extra_data, None

        df_d = df_d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # Variazione % Live del giorno
        fast_info = getattr(tk, 'fast_info', {})
        last_price = fast_info.get('lastPrice', float(df_d['Close'].iloc[-1]))
        prev_close = fast_info.get('previousClose', float(df_d['Close'].iloc[-2] if len(df_d) >= 2 else last_price))
        
        pct_change_today = ((last_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        extra_data['daily_var_pct'] = pct_change_today

        # Inferenza TimesFM Daily (orizzonte 5 giorni lavorativi)
        d_change_pct, d_target_price = predictor.predict_change(df_d['Close'], horizon=5)
        score_d = convert_forecast_to_score(d_change_pct)
        
        stop_loss_d = last_price * (1.0 - (abs(d_change_pct) / 100.0 if d_change_pct < 0 else 0.02))
        signals_daily.append(f"🤖 TimesFM 5D Forecast: {d_change_pct:+.2f}% -> Target: {d_target_price:.2f}")
        signals_daily.append(format_trend_line(d_change_pct, d_target_price, stop_loss_d))

        # --- 2. PREVISIONE INTRADAY / HOURLY (1H) ---
        score_h = 0.5
        df_h = tk.history(period="1mo", interval="1h", auto_adjust=True)
        if not df_h.empty and len(df_h) >= 32:
            df_h = df_h[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            # Inferenza TimesFM Hourly (orizzonte 6 ore di contrattazione)
            h_change_pct, _ = predictor.predict_change(df_h['Close'], horizon=6)
            score_h = convert_forecast_to_score(h_change_pct)

        return signals_daily, score_d, score_h, extra_data, df_d

    except Exception as e:
        print(f"❌ Errore durante l'analisi TimesFM di {ticker}: {e}")
        return [], 0.5, 0.5, extra_data, None


def create_daily_report_section(
    title: str, 
    results: List[Tuple[str, List[str], float, float, Dict, Optional[pd.DataFrame]]], 
    descriptions: Dict
) -> str:
    if not results:
        return f"{title}\nNessun dato disponibile."
    
    # Ordinamento primario per Score Daily (score_d) decrescente
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = [f"{title}\n"]
    
    for ticker, _, score_d, score_h, extra_data, _ in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet_d = get_bullet(score_d)
        bullet_h = get_bullet(score_h)
        var_pct = extra_data.get('daily_var_pct', 0.0)
        sign = "+" if var_pct > 0 else ""
        
        url = f"https://antoniotonti.github.io/agente_borsa/forecast_etf/{ticker}.html"
        
        # Intestazione con Ticker e Variazione %
        header_line = f"🔹 [{ticker}]({url}) - {desc} ({sign}{var_pct:.2f}%)"
        # Prima riga sotto-livello: Rating Daily (TimesFM)
        daily_line = f"├ 📈 *1D Daily:* {bullet_d} Score: `{score_d:.3f}`"
        # Seconda riga sotto-livello: Rating Hourly (TimesFM)
        hourly_line = f"└ ⚡ *1H Intraday:* {bullet_h} Score: `{score_h:.3f}`\n"
        
        lines.extend([header_line, daily_line, hourly_line])
        
    return "\n".join(lines)


def create_portfolio_daily_report(results: List[Tuple[str, List[str], float, float, Dict, Optional[pd.DataFrame]]], descriptions: Dict) -> str:
    now_str = datetime.now().strftime("%H:%M")
    return create_daily_report_section(f"🤖 *FORECAST ETF PORTAFOGLIO ({now_str})*", results, descriptions)


def create_watchlist_daily_report(results: List[Tuple[str, List[str], float, float, Dict, Optional[pd.DataFrame]]], descriptions: Dict) -> str:
    now_str = datetime.now().strftime("%H:%M")
    return create_daily_report_section(f"👁️ *FORECAST ETF OSSERVATI ({now_str})*", results, descriptions)


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
        print("🤖 AGENTE TRADING - FORECAST ETF (GOOGLE TIMESFM)")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        # Inizializzazione predittore TimesFM
        predictor = TimesFMPredictor()
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        
        portfolio_results = []
        if portfolio:
            print("\n💰 ANALISI PORTAFOGLIO ETF")
            for ticker in portfolio:
                signals_d, score_d, score_h, extra_data, df_d = analyze_etf_timesfm(ticker, predictor)
                portfolio_results.append((ticker, signals_d, score_d, score_h, extra_data, df_d))
                if df_d is not None and not df_d.empty:
                    desc = descriptions.get(ticker, ticker)
                    generate_web_page(ticker, desc, "forecast_etf", df_d, score_d, signals_d)
                time.sleep(0.3)
                
        watchlist_results = []
        if watchlist:
            print("\n👁️ ANALISI WATCHLIST ETF")
            for ticker in watchlist:
                signals_d, score_d, score_h, extra_data, df_d = analyze_etf_timesfm(ticker, predictor)
                watchlist_results.append((ticker, signals_d, score_d, score_h, extra_data, df_d))
                if df_d is not None and not df_d.empty:
                    desc = descriptions.get(ticker, ticker)
                    generate_web_page(ticker, desc, "forecast_etf", df_d, score_d, signals_d)
                time.sleep(0.3)
                
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

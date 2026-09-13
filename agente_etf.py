#!/usr/bin/env python3
"""
Agente di Trading - Analisi ETF
Legge SOLO i titoli con tipo == "ETF" da titoli.csv.
Motore di scoring con kill-switch ADX per fasi laterali.
"""

import os
import sys
from datetime import datetime

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta

sys.path.append('.')
from analysis_utils import calculate_heikin_ashi


# ==========================================
# CONFIGURAZIONE
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CSV_PATH = "titoli.csv"


# ==========================================
# CARICAMENTO ETF DA CSV
# ==========================================
def load_etf_from_csv(csv_path: str = CSV_PATH):
    """
    Legge titoli.csv e restituisce:
      - lista_etf: lista di ticker con tipo == 'ETF'
      - descriptions: dict {ticker: descrizione}
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ File {csv_path} non trovato")
        return [], {}
    except Exception as e:
        print(f"❌ Errore lettura CSV: {e}")
        return [], {}

    # Normalizza i nomi delle colonne
    df.columns = [c.strip().lower() for c in df.columns]

    if 'tipo' not in df.columns or 'codice' not in df.columns:
        print("❌ Il CSV deve avere le colonne 'codice' e 'tipo'")
        return [], {}

    # Filtra solo ETF (case-insensitive)
    etf_df = df[df['tipo'].astype(str).str.strip().str.upper() == "ETF"]

    lista_etf = etf_df['codice'].astype(str).str.strip().tolist()

    descriptions = {}
    if 'descrizione' in df.columns:
        for _, row in df.iterrows():
            descriptions[str(row['codice']).strip()] = str(row['descrizione']).strip()

    print(f"✅ Trovati {len(lista_etf)} ETF nel CSV: {lista_etf}")
    return lista_etf, descriptions


# ==========================================
# MOTORE DI SCORING CENTRALIZZATO
# ==========================================
def analyze_df_engine(df: pd.DataFrame) -> float:
    """
    Calcola gli indicatori tecnici e genera uno score da 0.0 a 1.0.
    Include il kill-switch ADX per fasi laterali.
    """
    if df.empty or len(df) < 35:
        return 0.0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy().dropna()

    if len(df) < 35:
        return 0.0

    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    # --- Indicatori (libreria `ta`) ---
    ema10 = ta.trend.ema_indicator(close, window=10)
    sma31 = ta.trend.sma_indicator(close, window=31)
    rsi = ta.momentum.rsi(close, window=14)

    macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = macd_obj.macd_diff()

    adx_obj = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    adx = adx_obj.adx()

    ha = calculate_heikin_ashi(df)

    try:
        ema_now = float(ema10.dropna().iloc[-1])
        sma_now = float(sma31.dropna().iloc[-1])
        rsi_val = float(rsi.dropna().iloc[-1])
        macd_h = float(macd_hist.dropna().iloc[-1])
        adx_val = float(adx.dropna().iloc[-1])
        ha_close = float(ha['HA_Close'].iloc[-1])
        ha_open = float(ha['HA_Open'].iloc[-1])
    except (IndexError, ValueError) as e:
        print(f"   ⚠️ Dati insufficienti per gli indicatori: {e}")
        return 0.0

    # Kill-Switch ADX (fase laterale → score azzerato)
    if pd.isna(adx_val) or adx_val < 20:
        return 0.0

    score = 0.0

    if ema_now > sma_now:
        score += 0.25
    if 50 < rsi_val < 70:
        score += 0.25
    if macd_h > 0:
        score += 0.25
    if ha_close > ha_open:
        score += 0.25

    return min(score, 1.0)


# ==========================================
# ANALISI MULTI-TIMEFRAME & RISCHIO
# ==========================================
def analyze_flash_ticker(ticker: str):
    try:
        df_1d = yf.download(ticker, period="6mo", interval="1d",
                            auto_adjust=True, progress=False)
        df_1h = yf.download(ticker, period="1mo", interval="1h",
                            auto_adjust=True, progress=False)

        if df_1d.empty or df_1h.empty:
            print(f"   ⚠️ {ticker}: dati insufficienti (1D o 1H vuoti)")
            return None

        if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = df_1d.columns.get_level_values(0)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)

        score_1d = analyze_df_engine(df_1d)
        score_1h = analyze_df_engine(df_1h)

        # --- Gestione Rischio (ATR Daily) ---
        df_1d = df_1d[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        atr_obj = ta.volatility.AverageTrueRange(
            high=df_1d['High'].squeeze(),
            low=df_1d['Low'].squeeze(),
            close=df_1d['Close'].squeeze(),
            window=14,
        )
        atr_series = atr_obj.average_true_range().dropna()

        if atr_series.empty:
            print(f"   ⚠️ {ticker}: ATR non calcolabile")
            return None

        atr = float(atr_series.iloc[-1])
        current_price = float(df_1d['Close'].iloc[-1])
        prev_price = float(df_1d['Close'].iloc[-2])
        daily_pct_change = ((current_price - prev_price) / prev_price) * 100.0

        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 3.0)
        sizing_risk = (atr * 1.5) / current_price * 100.0

        return {
            "ticker": ticker,
            "price": round(current_price, 3),
            "pct_change": round(daily_pct_change, 2),
            "score_1d": round(score_1d, 2),
            "score_1h": round(score_1h, 2),
            "atr": round(atr, 3),
            "stop_loss": round(stop_loss, 3),
            "take_profit": round(take_profit, 3),
            "sizing_risk_pct": round(sizing_risk, 2),
        }

    except Exception as e:
        print(f"   ❌ Errore su {ticker}: {e}")
        return None


# ==========================================
# GESTIONE MESSAGGI TELEGRAM
# ==========================================
def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti!")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"❌ Errore Telegram ({resp.status_code}): {resp.text}")
            return False
        print("✅ Messaggio Telegram inviato")
        return True
    except Exception as e:
        print(f"❌ Eccezione invio Telegram: {e}")
        return False
def get_volatility_bullet(sizing_risk_pct: float) -> tuple:
    """
    Restituisce (pallino, etichetta) in base alla Size Rischio (%).
    Size Rischio = distanza dello SL in % dal prezzo → proxy volatilità.
    """
    if sizing_risk_pct < 2.0:
        return "🟢", "Bassa"
    elif sizing_risk_pct <= 4.0:
        return "⚪", "Media"
    else:
        return "🔴", "Alta"

def format_telegram_alert(title: str, results: list, descriptions: dict) -> str:
    """Formatta l'output per Telegram ordinato per Score Daily."""
    if not results:
        return f"<b>{title}</b>\nNessun segnale rilevante."

    sorted_results = sorted(
        results,
        key=lambda x: (x['score_1d'], x['score_1h']),
        reverse=True,
    )

    RISK_PER_TRADE_PCT = 2.0  # Rischio fisso per trade

    msg = f"<b>{title}</b>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"

    for r in sorted_results:
        if r['score_1d'] >= 0.75:
            trend_emoji = "🟢"
        elif r['score_1d'] >= 0.5:
            trend_emoji = "⚪"
        else:
            trend_emoji = "🔴"

        change_sign = "+" if r['pct_change'] > 0 else ""
        desc = descriptions.get(r['ticker'], r['ticker'])

        # Volatilità (basata su Size Rischio)
        vol_bullet, vol_label = get_volatility_bullet(r['sizing_risk_pct'])

        # Calcolo posizione massima sul capitale (rischio fisso 3%)
        if r['sizing_risk_pct'] > 0:
            max_capital_pct = (RISK_PER_TRADE_PCT / r['sizing_risk_pct']) * 100.0
            max_capital_pct = min(max_capital_pct, 100.0)  # cap a 100%
        else:
            max_capital_pct = 100.0

        msg += f"{trend_emoji} <b>{r['ticker']}</b> - {desc} | {r['price']:.3f}$ ({change_sign}{r['pct_change']:.2f}%)\n"
        msg += f"   ├ Score 1D: {r['score_1d']} | Score 1H: {r['score_1h']}\n"
        msg += f"   ├ Rischio: SL {r['stop_loss']:.3f}$ | TP {r['take_profit']:.3f}$\n"
        msg += f"   └ {vol_bullet} Volatilità: {vol_label} (ATR {r['atr']:.3f}) | Size Rischio: {r['sizing_risk_pct']:.2f}%\n"
        msg += f"      💰 Rischio {RISK_PER_TRADE_PCT:.0f}% → max <b>{max_capital_pct:.0f}%</b> del capitale\n\n"

    return msg
# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("📊 AGENTE ETF - ANALISI MULTI-TIMEFRAME")
    print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)

    # Carica gli ETF dal CSV
    lista_etf, descriptions = load_etf_from_csv(CSV_PATH)

    if not lista_etf:
        print("⚠️ Nessun ETF trovato nel CSV. Uscita.")
        return

    # Analizza ogni ETF
    results = []
    print(f"\n💰 ANALISI {len(lista_etf)} ETF")
    for ticker in lista_etf:
        print(f"   → {ticker}")
        res = analyze_flash_ticker(ticker)
        if res:
            results.append(res)

    # Invio Telegram
    if results:
        print(f"\n📩 Invio alert ETF ({len(results)} strumenti)...")
        msg = format_telegram_alert("📊 ANALISI FLASH: ETF", results, descriptions)
        send_telegram_message(msg)
    else:
        print("\n⚠️ Nessun risultato valido, nessun invio.")

    print(f"\n🏁 Completato: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()

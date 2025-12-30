import os
from datetime import datetime
import requests
import yfinance as yf
import ta
import pandas as pd
import numpy as np

DESCR = {
    "STM": "STMicroelectronics N.V.",
    "SPM.MI": "Saipem S.p.A.",
    "AMP.MI": "Amplifon S.p.A.",
    "ZGN.MI": "Zignago Vetro S.p.A.",
    "NEXI.MI": "Nexi S.p.A.",
    "TIT.MI": "Telecom Italia S.p.A.",
    "BSS.MI": "Biesse S.p.A.",
    "TSL.MI": "Tessellis S.p.A.",
    "PRY.MI": "Prysmian S.p.A.",
    "REC.MI": "Recordati S.p.A.",
    "WBD.MI": "Webuild S.p.A.",
    "CPR.MI": "Campari S.p.A.",
    "FCT.MI": "Fincantieri S.p.A.",
    "PIRC.MI": "Pirelli S.p.A."
}

def load_portfolio():
    with open("portfolio.txt") as f:
        return [line.strip().upper() for line in f if line.strip()]

def heikin_ashi_color(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open  = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    return "bull" if ha_close.iloc[-1].item() > ha_open.iloc[-1].item() else "bear"

def zigzag_direction(df, depth=10):
    close = df['Close'].squeeze()
    recent = close.tail(depth)
    if close.iloc[-1] > recent.max() * 0.98: return "up"
    if close.iloc[-1] < recent.min() * 1.02: return "down"
    return "flat"

def check_signals(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if len(df) < 50: return []
        close = df['Close'].squeeze()
        ma31  = ta.trend.sma_indicator(close, window=31)
        ema10 = ta.trend.ema_indicator(close, window=10)
        crossover = np.where(ma31 > ema10, 1, 0)
        signal = pd.Series(crossover).diff().iloc[-1]
        zz_dir = zigzag_direction(df)
        color_prev = heikin_ashi_color(df.iloc[:-1])
        color_now  = heikin_ashi_color(df)
        alerts = []
        if signal == 1:  alerts.append("🟢 Incrocio rialzista MA31/EMA10")
        if signal == -1: alerts.append("🔴 Incrocio ribassista MA31/EMA10")
        if zz_dir != "flat": alerts.append(f"📊 ZigZag: {zz_dir.upper()}")
        if color_now != color_prev: alerts.append(f"🕯️ Heikin Ashi: {color_now.upper()}")
        return alerts
    except Exception as e:
        print(f"❌ {ticker}: {e}")
        return []

# ---------- MAIN ----------
if __name__ == "__main__":
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    print(f"DEBUG: token presente = {bool(token)}")
    print(f"DEBUG: chat_id presente = {bool(chat_id)}")

    segnali = []
    for ticker in load_portfolio():
        alerts = check_signals(ticker)
        if alerts:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} 📬 {ticker}: {' | '.join(alerts)}")
            segnali.append(f"*{ticker}* – {DESCR.get(ticker, ticker)}\n" + "\n".join(alerts))

    print(f"DEBUG: segnali totali {len(segnali)}")
    if segnali and token and chat_id:
        text = f"📈 Segnali Borsa {datetime.now().strftime('%d/%m %H:%M')}\n\n" + "\n\n".join(segnali)
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        print(f"DEBUG Telegram: {resp.status_code} - {resp.text}")
    else:
        print("DEBUG: nessun invio (mancano segnali o token/chat)")

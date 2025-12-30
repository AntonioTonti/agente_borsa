import os
from datetime import datetime
import requests
import yfinance as yf
import ta
import json
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import inspect, os
print("=== CODICE ESEGUITO ===")
with open(__file__, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if "sendMessage" in line:
            print(f"{i:03d}: {line.rstrip()}")
print("=== FINE ===")


EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = "antonio.tonti@tiscali.it"

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
    "CPR.MI": "Davide Campari-Milano S.p.A.",
    "FCT.MI": "Fincantieri S.p.A.",
    "PIRC.MI": "Pirelli & C. S.p.A."
}

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

def load_portfolio():
    with open("portfolio.txt") as f:
        return [line.strip().upper() for line in f if line.strip()]

def heikin_ashi_color(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open  = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    bull = ha_close.iloc[-1].item() > ha_open.iloc[-1].item()
    return "bull" if bull else "bear"

def zigzag_direction(df, depth=10):
    close = df['Close'].squeeze()
    recent = close.tail(depth)
    if close.iloc[-1] > recent.max() * 0.98:
        return "up"
    elif close.iloc[-1] < recent.min() * 1.02:
        return "down"
    return "flat"

def check_signals(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            print(f"⚠️ Dati insufficienti per {ticker}")
            return

        close = df['Close'].squeeze()

        # MA 31 vs EMA 10
        ma31 = ta.trend.sma_indicator(close, window=31)
        ema10 = ta.trend.ema_indicator(close, window=10)
        crossover = np.where(ma31 > ema10, 1, 0)
        signal = pd.Series(crossover).diff().iloc[-1]

        # ZigZag
        zz_dir = zigzag_direction(df)

        # Heikin Ashi
        color_prev = heikin_ashi_color(df.iloc[:-1])
        color_now = heikin_ashi_color(df)

        alerts = []
        if signal == 1:
            alerts.append("🟢 Incrocio rialzista MA31/EMA10")
        elif signal == -1:
            alerts.append("🔴 Incrocio ribassista MA31/EMA10")

        if zz_dir != "flat":
            alerts.append(f"📊 ZigZag cambio direzione: {zz_dir.upper()}")

        if color_now != color_prev:
            alerts.append(f"🕯️ Heikin Ashi cambio colore: {color_now.upper()}")

        if alerts:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M')} 📬 SEGNALI per {ticker}:")
            print("\n".join(alerts))
            # aggiunge a lista globale
            SEGNALI_ODIERNI.append({
                "ticker": ticker,
                "descr": DESCR.get(ticker, ticker),
                "direzione": "up" if "rialzista" in alerts[0] or "UP" in alerts[0] else "down",
                "testi": alerts
            })
    except Exception as e:
        print(f"❌ Errore su {ticker}: {e}")
# ---------- inizializza lista globale ----------
# ---------- inizializza lista globale ----------
SEGNALI_ODIERNI = []

if __name__ == "__main__":
    portfolio = load_portfolio()
    for ticker in portfolio:
        check_signals(ticker)

print(f"DEBUG: token presente = {bool(TOKEN)}")
print(f"DEBUG: chat_id presente = {bool(CHAT_ID)}")

    
    print(f"DEBUG: segnali trovati {len(SEGNALI_ODIERNI)}")

    # invio su Telegram
    if SEGNALI_ODIERNI:
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        testo = f"📈 Segnali Borsa {datetime.now().strftime('%d/%m %H:%M')}\n\n"
        for s in SEGNALI_ODIERNI:
            testo += f"*{s['ticker']}* – {s['descr']}\n"
            testo += "\n".join(s['testi']) + "\n\n"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": testo, "parse_mode": "Markdown"}
        resp = requests.post(url, data=payload, timeout=10)
        print(f"DEBUG Telegram: status={resp.status_code}, text={resp.text}")




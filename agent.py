import yfinance as yf
import ta
import pandas as pd
import numpy as np
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Config
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = "antonio.tonti@tiscali.it"

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
        return [line.strip() for line in f if line.strip()]

def heikin_ashi_color(df):
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    return "bull" if ha_close.iloc[-1] > ha_open.iloc[-1] else "bear"

def zigzag_direction(df, deviation=10, depth=10):
    close = df['Close'].squeeze()
    zz = ta.trend.zigzag(close, deviation=deviation, depth=depth)
    if zz.iloc[-1] > zz.iloc[-2]:
        return "up"
    elif zz.iloc[-1] < zz.iloc[-2]:
        return "down"
    return "flat"

def check_signals(ticker):
    df = yf.download(ticker, period="3mo", interval="1d")
    if len(df) < 50:
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
    color_prev = heikin_ashi_color(df[:-1])
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
        send_email(f"[{ticker}] Segnali attivi", "\n".join(alerts))

if __name__ == "__main__":
    portfolio = load_portfolio()
    for ticker in portfolio:
        check_signals(ticker)

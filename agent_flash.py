#!/usr/bin/env python3
import os, sys, time
from datetime import datetime
import requests
import yfinance as yf

sys.path.append('.')
from config import load_titoli_csv
from analysis_utils import calculate_heikin_ashi, get_bullet, calculate_trend_estimate, format_trend_line, ta
from state_manager import load_previous_state, save_current_state, calculate_deltas
from web_generator import generate_web_page

PERIOD = "1mo"
INTERVAL = "1h"
MIN_POINTS = 20
GITHUB_USER_BASE = "https://antoniotonti.github.io/agente_borsa/flash"

def analyze_ticker(ticker: str, desc: str):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty or len(df) < MIN_POINTS:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].squeeze()
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    var_pct = ((last_close - prev_close) / prev_close) * 100.0

    signals = []
    score = 0.5

    # Heikin Ashi
    ha = calculate_heikin_ashi(df)
    if len(ha) >= 2:
        if ha['HA_Close'].iloc[-1] > ha['HA_Open'].iloc[-1]:
            signals.append("🟢 HA: Barra Verde")
            score += 0.20
        else:
            signals.append("🔴 HA: Barra Rossa")
            score -= 0.20

    # Trend 3 periodi
    if len(close) >= 6:
        v, t, s = calculate_trend_estimate(close, lookback=3)
        signals.append(format_trend_line(v, t, s))

    # RSI
    rsi = ta.momentum.rsi(close, window=14).dropna()
    if not rsi.empty:
        signals.append(f"📊 RSI: {rsi.iloc[-1]:.1f}")

    score = max(0.0, min(1.0, score))
    
    # Generazione pagina web
    web_url = generate_web_page(ticker, desc, "flash", df, score, signals)

    return {
        "ticker": ticker,
        "desc": desc,
        "var_pct": round(var_pct, 2),
        "score": round(score, 3),
        "web_url": web_url
    }

def build_telegram_section(title: str, items: list, prev_state: dict, current_state_acc: dict) -> str:
    if not items:
        return ""
    
    lines = [f"⚡ *{title}*\n"]
    for item in items:
        t = item['ticker']
        d = item['desc']
        v = item['var_pct']
        s = item['score']
        url = item['web_url']
        
        current_state_acc[t] = {"var_pct": v, "score": s}
        
        d_var, d_score = calculate_deltas(t, v, s, prev_state)
        
        sign_v = "+" if v >= 0 else ""
        icon_v = "🟢" if v >= 0 else "🔴"
        sign_dv = "+" if d_var >= 0 else ""
        sign_ds = "+" if d_score >= 0 else ""

        # Formattazione 3 Righe
        line1 = f"{icon_v} [{t} - {d}]({url}) *[{sign_v}{v:.2f}%]* | *Score: {s:.3f}*"
        line2 = f"📊 Delta: Var *{sign_dv}{d_var:.2f}%* | Score *{sign_ds}{d_score:.3f}*"
        line3 = "⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯"
        
        lines.extend([line1, line2, line3])
        
    return "\n".join(lines)

def main():
    portfolio, watchlist, descriptions = load_titoli_csv()
    prev_state = load_previous_state("flash")
    current_state = {}

    p_items = [res for t in portfolio if (res := analyze_ticker(t, descriptions.get(t, t)))]
    w_items = [res for t in watchlist if (res := analyze_ticker(t, descriptions.get(t, t)))]

    # Ordina per score
    p_items.sort(key=lambda x: x['score'], reverse=True)
    w_items.sort(key=lambda x: x['score'], reverse=True)

    msg_p = build_telegram_section("FLASH PORTAFOGLIO", p_items, prev_state, current_state)
    msg_w = build_telegram_section("FLASH WATCHLIST", w_items, prev_state, current_state)

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        for msg in [msg_p, msg_w]:
            if msg:
                requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True})

    save_current_state("flash", current_state)

if __name__ == "__main__":
    main()

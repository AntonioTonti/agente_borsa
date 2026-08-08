#!/usr/bin/env python3
import os, sys
import requests
import yfinance as yf
import pandas as pd

sys.path.append('.')
from config import load_titoli_csv
from analysis_utils import calculate_heikin_ashi, calculate_trend_estimate, format_trend_line, ta
from state_manager import load_previous_state, save_current_state, calculate_deltas
from web_generator import generate_web_page

PERIOD = "1y"
INTERVAL = "1wk"
MIN_POINTS = 20

def analyze_ticker(ticker: str, desc: str):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty or len(df) < MIN_POINTS:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].squeeze()
    var_pct = ((float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2])) * 100.0

    signals = []
    score = 0.5

    ha = calculate_heikin_ashi(df)
    if len(ha) >= 2:
        if ha['HA_Close'].iloc[-1] > ha['HA_Open'].iloc[-1]:
            signals.append("🟢 HA Settimanale Verde")
            score += 0.25
        else:
            signals.append("🔴 HA Settimanale Rossa")
            score -= 0.25

    if len(close) >= 6:
        v, t, s = calculate_trend_estimate(close, lookback=3)
        signals.append(format_trend_line(v, t, s))

    score = max(0.0, min(1.0, score))
    web_url = generate_web_page(ticker, desc, "chiusura", df, score, signals)

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
    
    lines = [f"🌆 *{title}*\n"]
    for item in items:
        t, d, v, s, url = item['ticker'], item['desc'], item['var_pct'], item['score'], item['web_url']
        current_state_acc[t] = {"var_pct": v, "score": s}
        d_var, d_score = calculate_deltas(t, v, s, prev_state)
        
        sign_v = "+" if v >= 0 else ""
        icon_v = "🟢" if v >= 0 else "🔴"
        sign_dv = "+" if d_var >= 0 else ""
        sign_ds = "+" if d_score >= 0 else ""

        lines.append(f"{icon_v} [{t} - {d}]({url}) *[{sign_v}{v:.2f}%]* | *Score: {s:.3f}*")
        lines.append(f"📊 Delta: Var *{sign_dv}{d_var:.2f}%* | Score *{sign_ds}{d_score:.3f}*")
        lines.append("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
        
    return "\n".join(lines)

def main():
    portfolio, watchlist, descriptions = load_titoli_csv()
    prev_state = load_previous_state("chiusura")
    current_state = {}

    p_items = [res for t in portfolio if (res := analyze_ticker(t, descriptions.get(t, t)))]
    w_items = [res for t in watchlist if (res := analyze_ticker(t, descriptions.get(t, t)))]

    p_items.sort(key=lambda x: x['score'], reverse=True)
    w_items.sort(key=lambda x: x['score'], reverse=True)

    msg_p = build_telegram_section("CHIUSURA PORTAFOGLIO", p_items, prev_state, current_state)
    msg_w = build_telegram_section("CHIUSURA WATCHLIST", w_items, prev_state, current_state)

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        for msg in [msg_p, msg_w]:
            if msg:
                requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True})

    save_current_state("chiusura", current_state)

if __name__ == "__main__":
    main()

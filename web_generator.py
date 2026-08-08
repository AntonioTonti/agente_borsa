#!/usr/bin/env python3
import os
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from analysis_utils import calculate_heikin_ashi

BASE_DOCS_DIR = "docs"

def generate_web_page(ticker: str, desc: str, agent_type: str, df: pd.DataFrame, score: float, signals: list) -> str:
    agent_dir = agent_type.lower()
    out_dir = os.path.join(BASE_DOCS_DIR, agent_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    file_name = f"{ticker}.html"
    file_path = os.path.join(out_dir, file_name)

    # Dati da yfinance per Fondamentali & Analisti
    ticker_obj = yf.Ticker(ticker)
    info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        pass
    
    target_mean = info.get("targetMeanPrice", "N/D")
    target_high = info.get("targetHighPrice", "N/D")
    target_low = info.get("targetLowPrice", "N/D")
    rec_key = info.get("recommendationKey", "N/D").upper() if info.get("recommendationKey") else "N/D"
    pe_ratio = info.get("trailingPE", "N/D")
    market_cap = info.get("marketCap", "N/D")
    
    if isinstance(market_cap, (int, float)):
        market_cap = f"{market_cap / 1e9:.2f}B USD"

    # Calcolo Heikin Ashi e Indicatori per il Grafico
    ha_df = calculate_heikin_ashi(df)
    close = df['Close'].squeeze()
    ema10 = ta.trend.ema_indicator(close, window=10)
    ma31 = ta.trend.sma_indicator(close, window=31)
    rsi = ta.momentum.rsi(close, window=14)

    # Creazione Grafico Interattivo Plotly
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

    # Candele Heikin Ashi
    fig.add_trace(go.Candlestick(
        x=ha_df.index, open=ha_df['HA_Open'], high=ha_df['HA_High'],
        low=ha_df['HA_Low'], close=ha_df['HA_Close'], name="Heikin Ashi"
    ), row=1, col=1)

    # Medie Mobili
    fig.add_trace(go.Scatter(x=df.index, y=ema10, line=dict(color='orange', width=1.5), name="EMA 10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ma31, line=dict(color='blue', width=1.5), name="MA 31"), row=1, col=1)

    # Volumi
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'].squeeze(), name="Volumi", marker_color='gray'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color='purple', width=1.5), name="RSI (14)"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(title=f"{ticker} - {desc} ({agent_type.upper()})", template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
    chart_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

    signals_html = "".join([f"<li>{s}</li>" for s in signals])

    # Template HTML con Bootstrap Dark
    html_content = f"""<!DOCTYPE html>
<html lang="it" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} - Analisi {agent_type.upper()}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-dark text-light p-3">
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h2>{ticker} <small class="text-muted">({desc})</small></h2>
            <span class="badge bg-primary fs-5">Score: {score:.3f}</span>
        </div>

        <div class="row mb-3">
            <div class="col-md-3"><div class="card p-2"><strong>Rating Analisti:</strong> {rec_key}</div></div>
            <div class="col-md-3"><div class="card p-2"><strong>Target Medio:</strong> {target_mean}</div></div>
            <div class="col-md-3"><div class="card p-2"><strong>Target Range:</strong> {target_low} - {target_high}</div></div>
            <div class="col-md-3"><div class="card p-2"><strong>P/E Ratio:</strong> {pe_ratio} | <strong>Cap:</strong> {market_cap}</div></div>
        </div>

        <div class="card p-2 mb-3">
            {chart_html}
        </div>

        <div class="card p-3">
            <h4>Segnali Rilevati</h4>
            <ul>{signals_html}</ul>
        </div>
    </div>
</body>
</html>"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return f"https://antoniotonti.github.io/agente_borsa/{agent_dir}/{file_name}"

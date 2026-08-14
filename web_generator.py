#!/usr/bin/env python3
"""
Modulo per la generazione della pagina HTML con grafico Plotly avanzato.
Ottimizzato per candele orarie (asse X discreto per eliminare buchi notturni e weekend).
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_web_page(ticker: str, desc: str, mode: str, df: pd.DataFrame, score: float, signals: list):
    """
    Genera la pagina HTML per il singolo titolo con grafico interattivo.
    """
    try:
        output_dir = f"docs/{mode}"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{ticker}.html")

        # Formattazione data/ora per l'asse X discreto (elimina buchi di notte e weekend)
        if isinstance(df.index, pd.DatetimeIndex):
            x_dates = df.index.strftime('%d/%m %H:%M')
        else:
            x_dates = [str(x) for x in df.index]

        # Colori per i volumi (Verdi se Close >= Open, Rossi se Close < Open)
        vol_colors = ['#00c853' if c >= o else '#ff5252' for c, o in zip(df['Close'], df['Open'])]

        # Creazione figura con subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=(f"{ticker} - {desc} ({mode.upper()})", "", "")
        )

        # 1. Candele
        fig.add_trace(
            go.Candlestick(
                x=x_dates,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Prezzo",
                increasing_line_color='#00c853',
                decreasing_line_color='#ff5252',
                increasing_fillcolor='#00c853',
                decreasing_fillcolor='#ff5252'
            ),
            row=1, col=1
        )

        # 2. Volumi (Barre colorate e piene)
        fig.add_trace(
            go.Bar(
                x=x_dates,
                y=df['Volume'],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.75
            ),
            row=2, col=1
        )

        # 3. Indicatori extra se presenti (es. RSI o Medie)
        if 'EMA10' in df.columns:
            fig.add_trace(go.Scatter(x=x_dates, y=df['EMA10'], mode='lines', name='EMA 10', line=dict(color='#ffa726', width=1.5)), row=1, col=1)
        if 'MA31' in df.columns:
            fig.add_trace(go.Scatter(x=x_dates, y=df['MA31'], mode='lines', name='MA 31', line=dict(color='#29b6f6', width=1.5)), row=1, col=1)

        # Layout Dark Mode
        fig.update_layout(
            template="plotly_dark",
            height=800,
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            paper_bgcolor="#121212",
            plot_bgcolor="#1e1e1e"
        )

        # Eliminazione spazi vuoti (notti/weekend) imponendo asse categoriale
        fig.update_xaxes(type='category', nticks=12)

        # Formattazione lista segnali
        signals_html = "".join([f"<li>{s}</li>" for s in signals])

        # Badge colore score
        if score >= 0.70:
            score_color = "#00c853"
        elif score <= 0.40:
            score_color = "#ff5252"
        else:
            score_color = "#9e9e9e"

        # HTML Template
        html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} - Analisi {mode.upper()}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; background: #1e1e1e; padding: 15px 25px; border-radius: 8px; margin-bottom: 20px; }}
        .score-badge {{ background-color: {score_color}; color: #fff; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 1.1em; }}
        .signals-card {{ background: #1e1e1e; padding: 20px; border-radius: 8px; margin-top: 20px; }}
        .signals-card ul {{ list-style-type: none; padding-left: 0; }}
        .signals-card li {{ padding: 6px 0; border-bottom: 1px solid #2c2c2c; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{ticker} ({desc})</h2>
        <div class="score-badge">Score: {score:.3f}</div>
    </div>
    <div id="plotly-chart"></div>
    <div class="signals-card">
        <h3>Segnali Rilevati</h3>
        <ul>{signals_html}</ul>
    </div>
    <script>
        var plotData = {fig.to_json()};
        Plotly.newPlot('plotly-chart', plotData.data, plotData.layout);
    </script>
</body>
</html>
"""

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 Pagina Web generata: {file_path}")

    except Exception as e:
        print(f"⚠️ Errore nella generazione della pagina web per {ticker}: {e}")

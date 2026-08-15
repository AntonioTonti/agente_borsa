#!/usr/bin/env python3
"""
Modulo per la generazione della pagina HTML con grafico Plotly avanzato.
Utilizza rangebreaks native per eliminare i buchi notturni/weekend senza rompere il rendering di Candlestick e Volumi.
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

        # Verifica e assicura l'indice Datetime
        plot_df = df.copy()
        if not isinstance(plot_df.index, pd.DatetimeIndex):
            plot_df.index = pd.to_datetime(plot_df.index)

        # Colori per i volumi (Verde se Close >= Open, Rosso se Close < Open)
        vol_colors = ['#00c853' if c >= o else '#ff5252' for c, o in zip(plot_df['Close'], plot_df['Open'])]

        # Creazione figura con 3 subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.60, 0.20, 0.20],
            subplot_titles=(f"{ticker} - {desc} ({mode.upper()})", "", "")
        )

        # 1. CANDELE PREZZO (RIGA 1)
        fig.add_trace(
            go.Candlestick(
                x=plot_df.index,
                open=plot_df['Open'],
                high=plot_df['High'],
                low=plot_df['Low'],
                close=plot_df['Close'],
                name="Prezzo",
                increasing_line_color='#00c853',
                decreasing_line_color='#ff5252',
                increasing_fillcolor='#00c853',
                decreasing_fillcolor='#ff5252'
            ),
            row=1, col=1
        )

        # 2. SOVRAPPOSIZIONE RIFERIMENTI DI PREZZO & INDICATORI (RIGA 1)
        # Supertrend / Trailing Stop (se presenti)
        if 'Supertrend' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Supertrend'], mode='lines', name='Supertrend', line=dict(color='#ffd54f', width=2, dash='dot')), row=1, col=1)
        
        # Medie Mobili
        if 'EMA10' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA10'], mode='lines', name='EMA 10', line=dict(color='#ffa726', width=1.5)), row=1, col=1)
        if 'MA31' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA31'], mode='lines', name='MA 31', line=dict(color='#29b6f6', width=1.5)), row=1, col=1)

        # Target Price e Stop Loss (se presenti nel DF)
        if 'Target' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Target'], mode='lines', name='Target Price', line=dict(color='#00e676', width=1.5, dash='dash')), row=1, col=1)
        if 'StopLoss' in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['StopLoss'], mode='lines', name='Stop Loss', line=dict(color='#ff1744', width=1.5, dash='dash')), row=1, col=1)

        # 3. VOLUMI (RIGA 2)
        fig.add_trace(
            go.Bar(
                x=plot_df.index,
                y=plot_df['Volume'],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.75
            ),
            row=2, col=1
        )

        # 4. RSI (RIGA 3)
        if 'RSI' in plot_df.columns:
            rsi_series = plot_df['RSI']
        else:
            import ta
            rsi_series = ta.momentum.rsi(plot_df['Close'], window=9 if mode == 'flash' else 14)

        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=rsi_series,
                mode='lines',
                name="RSI",
                line=dict(color='#ab47bc', width=1.5)
            ),
            row=3, col=1
        )

        # Linee di soglia RSI (30 e 70)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff5252", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00c853", row=3, col=1)

        # LAYOUT DARK MODE
        fig.update_layout(
            template="plotly_dark",
            height=800,
            margin=dict(l=40, r=40, t=50, b=40),
            showlegend=True,
            paper_bgcolor="#121212",
            plot_bgcolor="#1e1e1e"
        )

        # CONFIGURAZIONE ASSE X NATIVA CON RANGEBREAKS (Niente buchi, Candele cicciotte)
        fig.update_xaxes(
            rangeslider_visible=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),           # Nasconde i Fine Settimana
                dict(bounds=[17.5, 9], pattern="hour") # Nasconde la notte (dalle 17:30 alle 09:00)
            ]
        )

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

        print(f"📄 Pagina Web generata con successo: {file_path}")

    except Exception as e:
        print(f"⚠️ Errore nella generazione della pagina web per {ticker}: {e}")

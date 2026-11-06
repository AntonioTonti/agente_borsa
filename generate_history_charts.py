#!/usr/bin/env python3
"""
Generatore di grafici storici degli Score.
Estrae i dati dai DB, raggruppa per giorno (per nascondere i vecchi dati orari)
e usa le descrizioni dal CSV.
"""

import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import sys

sys.path.append('.')
from config import load_titoli_csv

BASE_DOCS_DIR = "docs"
KENZO_DB = "kenzo_history.db"
ETF_DB = "etf_history.db"

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40
MAX_DAYS_HISTORY = 60 


def get_data_from_db(db_path: str, categoria: str) -> pd.DataFrame:
    """Estrae le previsioni dal DB e tiene solo l'ultimo dato di ogni giorno."""
    if not os.path.exists(db_path):
        print(f"⚠️ DB non trovato: {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT ticker, timestamp_emissione, score 
            FROM previsioni 
            WHERE categoria = ? 
              AND timestamp_emissione >= datetime('now', '-{MAX_DAYS_HISTORY} days')
            ORDER BY timestamp_emissione ASC
        """
        df = pd.read_sql_query(query, conn, params=(categoria,))
        conn.close()
        
        if not df.empty:
            df['timestamp_emissione'] = pd.to_datetime(df['timestamp_emissione'])
            
            # --- MAGIA: Raggruppiamo per giorno e teniamo solo l'ultima rilevazione ---
            # Questo nasconde i vecchi dati orari e mostra un solo punto al giorno
            df['data'] = df['timestamp_emissione'].dt.date
            df = df.drop_duplicates(subset=['ticker', 'data'], keep='last')
            df = df.drop(columns=['data'])
            
        return df
    except Exception as e:
        print(f"❌ Errore lettura DB {db_path} ({categoria}): {e}")
        return pd.DataFrame()


def create_multi_chart_page(df: pd.DataFrame, title: str, filename: str, descriptions: dict) -> str:
    if df.empty:
        print(f"⚠️ Nessun dato per {title}. Salto la generazione.")
        return ""

    charts_html = ""
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values(by='timestamp_emissione')
        
        desc = descriptions.get(ticker, ticker)
        chart_title = f"<b>{ticker}</b> - {desc}"
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ticker_df['timestamp_emissione'],
            y=ticker_df['score'],
            mode='lines+markers',
            name=ticker,
            line=dict(width=2, color='#00d2ff'), 
            marker=dict(size=6, color='#00d2ff') 
        ))

        fig.add_hline(y=BUY_THRESHOLD, line_dash="dash", line_color="#00ff00", 
                      annotation_text="BUY (>0.60)", annotation_position="top left")
        fig.add_hline(y=SELL_THRESHOLD, line_dash="dash", line_color="#ff3333", 
                      annotation_text="SELL (<0.40)", annotation_position="bottom left")

        if len(ticker_df) == 1:
            data_unica = ticker_df['timestamp_emissione'].iloc[0]
            fig.update_xaxes(range=[data_unica - pd.Timedelta(hours=12), data_unica + pd.Timedelta(hours=12)])

        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=20)),
            xaxis_title="Data",
            yaxis_title="Score (0.0 - 1.0)",
            yaxis=dict(range=[0.0, 1.0]),
            template="plotly_dark",
            height=600, 
            margin=dict(l=40, r=40, t=60, b=40)
        )

        charts_html += f'<div class="chart-container">{fig.to_html(include_plotlyjs="cdn", full_html=False)}</div>'

    out_dir = os.path.join(BASE_DOCS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)
    
    html_content = f"""<!DOCTYPE html>
<html lang="it" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Storico Score</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #121212; color: white; padding: 20px; }}
        .chart-container {{ 
            background: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        }}
        h1 {{ margin-bottom: 30px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1>{title} - Storico Score</h1>
        {charts_html}
    </div>
</body>
</html>"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return f"https://antoniotonti.github.io/agente_borsa/{filename}"


def main():
    print("=" * 60)
    print("📊 GENERAZIONE GRAFICI STORICI (Multi-Chart Giornaliero)")
    print("=" * 60)

    _, _, descriptions, _ = load_titoli_csv()

    df_etf = get_data_from_db(ETF_DB, "ETF")
    link_etf = create_multi_chart_page(df_etf, "ETF", "storico_etf.html", descriptions)
    if link_etf: print(f"✅ Generato ETF: {link_etf}")

    df_oss = get_data_from_db(KENZO_DB, "WATCHLIST")
    link_oss = create_multi_chart_page(df_oss, "Osservati", "storico_osservati.html", descriptions)
    if link_oss: print(f"✅ Generato Osservati: {link_oss}")

    df_port = get_data_from_db(KENZO_DB, "PORTAFOGLIO")
    link_port = create_multi_chart_page(df_port, "Portafoglio", "storico_portafoglio.html", descriptions)
    if link_port: 
        print(f"✅ Generato Portafoglio: {link_port}")
    else:
        print("ℹ️ Nessun titolo in portafoglio, grafico non generato (o vuoto).")

    print("\n🏁 Completato.")


if __name__ == "__main__":
    main()

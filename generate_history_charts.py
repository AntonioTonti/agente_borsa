#!/usr/bin/env python3
"""
Generatore di grafici storici degli Score.
Estrae i dati dai DB (kenzo_history.db ed etf_history.db) e genera 3 pagine HTML.
"""

import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go

BASE_DOCS_DIR = "docs"
KENZO_DB = "kenzo_history.db"
ETF_DB = "etf_history.db"

# Soglie operative
BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40


def get_data_from_db(db_path: str, categoria: str) -> pd.DataFrame:
    """Estrae le previsioni dal DB per una specifica categoria."""
    if not os.path.exists(db_path):
        print(f"⚠️ DB non trovato: {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        # Estraiamo solo id, ticker, data e score
        query = """
            SELECT ticker, timestamp_emissione, score 
            FROM previsioni 
            WHERE categoria = ?
            ORDER BY timestamp_emissione ASC
        """
        df = pd.read_sql_query(query, conn, params=(categoria,))
        conn.close()
        
        if not df.empty:
            df['timestamp_emissione'] = pd.to_datetime(df['timestamp_emissione'])
        return df
    except Exception as e:
        print(f"❌ Errore lettura DB {db_path} ({categoria}): {e}")
        return pd.DataFrame()


def create_chart(df: pd.DataFrame, title: str, filename: str) -> str:
    """Crea il grafico interattivo e lo salva come HTML."""
    if df.empty:
        print(f"⚠️ Nessun dato per {title}. Salto la generazione.")
        return ""

    fig = go.Figure()

    # Aggiungi una linea per ogni ticker
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_df['timestamp_emissione'],
            y=ticker_df['score'],
            mode='lines+markers',
            name=ticker,
            line=dict(width=2),
            marker=dict(size=4)
        ))

    # Aggiungi linee orizzontali per BUY/SELL
    fig.add_hline(y=BUY_THRESHOLD, line_dash="dash", line_color="green", 
                  annotation_text="BUY (>0.60)", annotation_position="top left")
    fig.add_hline(y=SELL_THRESHOLD, line_dash="dash", line_color="red", 
                  annotation_text="SELL (<0.40)", annotation_position="bottom left")

    fig.update_layout(
        title=f"{title} - Storico Score",
        xaxis_title="Data/Ora",
        yaxis_title="Score (0.0 - 1.0)",
        yaxis=dict(range=[0.0, 1.0]), # Forza l'asse Y da 0 a 1
        template="plotly_dark",
        hovermode="x unified", # Mostra tutti i valori passando col mouse
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Salvataggio
    out_dir = os.path.join(BASE_DOCS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)
    
    # Genera HTML completo con CSS dark theme
    html_content = f"""<!DOCTYPE html>
<html lang="it" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Storico Score</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body {{ background-color: #121212; color: white; padding: 20px; }}</style>
</head>
<body>
    <div class="container-fluid">
        {fig.to_html(include_plotlyjs='cdn', full_html=False)}
    </div>
</body>
</html>"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return f"https://antoniotonti.github.io/agente_borsa/{filename}"


def main():
    print("=" * 60)
    print("📊 GENERAZIONE GRAFICI STORICI")
    print("=" * 60)

    # 1. ETF
    df_etf = get_data_from_db(ETF_DB, "ETF")
    link_etf = create_chart(df_etf, "ETF", "storico_etf.html")
    if link_etf: print(f"✅ Generato ETF: {link_etf}")

    # 2. OSSERVATI (Kenzo)
    df_oss = get_data_from_db(KENZO_DB, "WATCHLIST")
    link_oss = create_chart(df_oss, "Osservati", "storico_osservati.html")
    if link_oss: print(f"✅ Generato Osservati: {link_oss}")

    # 3. PORTAFOGLIO (Kenzo)
    df_port = get_data_from_db(KENZO_DB, "PORTAFOGLIO")
    link_port = create_chart(df_port, "Portafoglio", "storico_portafoglio.html")
    if link_port: print(f"✅ Generato Portafoglio: {link_port}")

    print("\n🏁 Completato.")


if __name__ == "__main__":
    main()

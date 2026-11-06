def create_chart(df: pd.DataFrame, title: str, filename: str) -> str:
    """Crea il grafico interattivo e lo salva come HTML."""
    if df.empty:
        print(f"⚠️ Nessun dato per {title}. Salto la generazione.")
        return ""

    fig = go.Figure()

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

    fig.add_hline(y=BUY_THRESHOLD, line_dash="dash", line_color="green", 
                  annotation_text="BUY (>0.60)", annotation_position="top left")
    fig.add_hline(y=SELL_THRESHOLD, line_dash="dash", line_color="red", 
                  annotation_text="SELL (<0.40)", annotation_position="bottom left")

    fig.update_layout(
        title=f"{title} - Storico Score",
        xaxis_title="Data/Ora",
        yaxis_title="Score (0.0 - 1.0)",
        yaxis=dict(range=[0.0, 1.0]),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

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
        
    # ⚠️ QUI LA CORREZIONE: tolto "docs/" dal link pubblico
    return f"https://antoniotonti.github.io/agente_borsa/{filename}"

def format_signals_by_weight(signals: List[str]) -> List[str]:
    """
    Ordina gli indicatori secondo la priorità dei pesi (dal 18% al 5%)
    e li restituisce formattati.
    """
    # Mappatura delle chiavi per ordinare gli indicatori in base ai pesi
    priority_order = [
        "EMA10",             # Peso 18%
        "Stima Trend",       # Peso 15%
        "Delta EMA10/MA31",  # Peso 12%
        "Espansione Corpo",  # Peso 15% (HA Forza)
        "HEIKIN ASHI:",      # Peso 10% (HA Stato)
        "ZIGZAG:",           # Peso 10%
        "Volume:",           # Peso 5%
        "Chiusura vs Prec:", # Peso 5%
        "RSI:",              # Peso 5%
        "MACD"               # Peso 5%
    ]
    
    def get_priority(signal_text: str) -> int:
        for index, key in enumerate(priority_order):
            if key in signal_text:
                return index
        return 99

    # Ordina la lista in base alla priorità definita
    sorted_signals = sorted(signals, key=get_priority)
    return [f"  {sig.strip()}" for sig in sorted_signals if sig.strip()]

def create_daily_report_section(title: str, results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    """
    Crea la sezione del report giornaliero applicando la nuova formattazione.
    """
    if not results:
        return f"{title} - Nessun segnale oggi"
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    lines = [f"{title}"]
    
    for ticker, signals, score, extra_data in sorted_results:
        desc = descriptions.get(ticker, ticker)
        bullet = get_bullet(score)
        
        # Recupera la variazione percentuale (salvata nei dati o ricavata da 'Chiusura vs Prec')
        var_pct = 0.0
        for sig in signals:
            if "Chiusura vs Prec:" in sig:
                try:
                    var_str = sig.split("Chiusura vs Prec:")[1].replace("%", "").strip()
                    var_pct = float(var_str)
                except ValueError:
                    pass
                break
        
        # Colore pallino per la variazione percentuale
        change_icon = "🟢" if var_pct >= 0 else "🔴"
        sign = "+" if var_pct > 0 else ""
        
        # Prima riga: Codice - Descrizione [% variazione] (Verde/Rossa) | Score
        header_line = f"\n*{ticker} - {desc} [{change_icon} {sign}{var_pct:.2f}%]* {bullet} (*score: {score:.3f}*)"
        lines.append(header_line)
        
        if signals:
            # Ordina e formatta gli indicatori per peso
            formatted_signals = format_signals_by_weight(signals)
            lines.extend(formatted_signals)
        else:
            lines.append("  📭 Nessun segnale rilevato")
            
        # Separatore riga orizzontale
        lines.append("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯")
        
    return "\n".join(lines)

def create_portfolio_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    return create_daily_report_section("💰 *PORTAFOGLIO GIORNALIERO*", results, descriptions)

def create_watchlist_daily_report(results: List[Tuple[str, List[str], float, Dict]], descriptions: Dict) -> str:
    return create_daily_report_section("👁️ *OSSERVATI GIORNALIERI*", results, descriptions)

#!/usr/bin/env python3
"""
Configurazione agente di trading
"""

import pandas as pd
from typing import Tuple, Dict, List

# ============================================================================
# COSTANTI DI CONFIGURAZIONE
# ============================================================================

# Parametri per analisi giornaliera
DAILY_PERIOD = "2mo"      # Ultimi 2 mesi
DAILY_INTERVAL = "1d"     # Dati giornalieri
DAILY_MIN_POINTS = 32     # Minimo punti per analisi

# Parametri per analisi settimanale
WEEKLY_PERIOD = "1y"      # Ultimo anno
WEEKLY_INTERVAL = "1wk"   # Dati settimanali
WEEKLY_MIN_POINTS = 30    # Minimo punti per analisi

# ============================================================================
# FUNZIONI DI CARICAMENTO DATI
# ============================================================================

def load_titoli_csv(csv_path: str = "titoli.csv") -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Carica titoli da CSV e restituisce:
    - Lista portafoglio
    - Lista watchlist
    - Dizionario descrizioni
    """
    try:
        df = pd.read_csv(csv_path)
        
        portfolio = df[df['tipo'] == 'PORTFOLIO']['codice'].tolist()
        watchlist = df[df['tipo'] == 'WATCHLIST']['codice'].tolist()
        
        descriptions = {}
        for _, row in df.iterrows():
            descriptions[row['codice']] = row['descrizione']
        
        print(f"✅ CSV caricato: {len(portfolio)} portfolio, {len(watchlist)} watchlist")
        return portfolio, watchlist, descriptions
        
    except FileNotFoundError:
        print(f"❌ File {csv_path} non trovato")
        return [], [], {}
    except Exception as e:
        print(f"❌ Errore caricamento CSV: {e}")
        return [], [], {}

# ============================================================================
# FUNZIONI DI CONFIGURAZIONE RACCOMANDAZIONI
# ============================================================================

def load_config(config_path: str = "config.txt") -> Dict[str, float]:
    """Carica soglie da file di configurazione"""
    thresholds = {
        'STRONG_SELL': 0.25,
        'SELL': 0.35,
        'WARNING': 0.45,
        'NEUTRAL': 0.55,
        'BUY': 0.65,
        'STRONG_BUY': 0.75
    }
    
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in thresholds:
                            thresholds[key] = float(value)
        print(f"✅ Config caricato da {config_path}")
    except FileNotFoundError:
        print(f"⚠️  File {config_path} non trovato, uso valori default")
    except Exception as e:
        print(f"⚠️  Errore caricamento config: {e}, uso valori default")
    
    return thresholds

def get_recommendation(score: float, thresholds: Dict[str, float]) -> Tuple[str, str]:
    """Determina raccomandazione basata su score e soglie"""
    if score < thresholds['STRONG_SELL']:
        return "🔴🔴 VENDI SUBITO", "STRONG_SELL"
    elif score < thresholds['SELL']:
        return "🔴 CONSIGLIA VENDITA", "SELL"
    elif score < thresholds['WARNING']:
        return "🟡 MONITORA ATTIVAMENTE", "WARNING"
    elif score < thresholds['NEUTRAL']:
        return "⚪ MANTIENI POSIZIONE", "NEUTRAL"
    elif score < thresholds['BUY']:
        return "🟢 CONSIGLIA ACQUISTO", "BUY"
    else:
        return "🟢🟢 FORTE ACQUISTO", "STRONG_BUY"

if __name__ == "__main__":
    # Test delle funzioni
    portfolio, watchlist, desc = load_titoli_csv()
    print(f"Portfolio: {portfolio}")
    print(f"Watchlist: {watchlist}")
    print(f"Descrizioni: {len(desc)} voci")
    
    thresholds = load_config()
    print(f"Soglie: {thresholds}")

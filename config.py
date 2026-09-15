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
DAILY_PERIOD = "2mo"
DAILY_INTERVAL = "1d"
DAILY_MIN_POINTS = 32

# Parametri per analisi settimanale
WEEKLY_PERIOD = "1y"
WEEKLY_INTERVAL = "1wk"
WEEKLY_MIN_POINTS = 30

# ============================================================================
# FUNZIONI DI CARICAMENTO DATI
# ============================================================================

def load_titoli_csv(csv_path: str = "titoli.csv"):
    """
    Carica titoli da CSV e restituisce:
    - lista portafoglio
    - lista watchlist
    - dizionario descrizioni
    - lista ETF
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]

        portfolio = df[df['tipo'].str.upper() == 'PORTAFOGLIO']['codice'].tolist()
        watchlist = df[df['tipo'].str.upper() == 'WATCHLIST']['codice'].tolist()
        etf_list = df[df['tipo'].str.upper() == 'ETF']['codice'].tolist()

        descriptions = {}
        for _, row in df.iterrows():
            descriptions[row['codice']] = row['descrizione']

        print(f"✅ CSV caricato: {len(portfolio)} portfolio, {len(watchlist)} watchlist, {len(etf_list)} ETF")
        return portfolio, watchlist, descriptions, etf_list

    except FileNotFoundError:
        print(f"❌ File {csv_path} non trovato")
        return [], [], {}, []
    except Exception as e:
        print(f"❌ Errore caricamento CSV: {e}")
        return [], [], {}, []


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
    portfolio, watchlist, desc, etf = load_titoli_csv()
    print(f"Portfolio: {portfolio}")
    print(f"Watchlist: {watchlist}")
    print(f"ETF: {etf}")
    print(f"Descrizioni: {len(desc)} voci")

    thresholds = load_config()
    print(f"Soglie: {thresholds}")

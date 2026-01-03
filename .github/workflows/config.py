"""
Configurazione centralizzata per l'agente di trading
"""

# ============================================================================
# FILE CONFIGURAZIONE
# ============================================================================

PORTFOLIO_FILE = "portfolio.txt"
WATCHLIST_FILE = "watchlist.txt"
CONFIG_FILE = "config.txt"

# ============================================================================
# SOGLIE SCORE (personalizzabili via config.txt)
# ============================================================================

DEFAULT_THRESHOLDS = {
    'STRONG_SELL': 0.25,
    'SELL': 0.35,
    'WARNING': 0.45,
    'NEUTRAL': 0.55,
    'BUY': 0.65,
    'STRONG_BUY': 0.75
}

# ============================================================================
# DESCRIZIONI TICKER
# ============================================================================

TICKER_DESCRIPTIONS = {
    "STM": "STMicroelectronics N.V.",
    "SPM.MI": "Saipem S.p.A.",
    "AMP.MI": "Amplifon S.p.A.",
    "ZV.MI": "Zignago Vetro S.p.A.",
    "NEXI.MI": "Nexi S.p.A.",
    "TIT.MI": "Telecom Italia S.p.A.",
    "BSS.MI": "Biesse S.p.A.",
    "TSL.MI": "Tessellis S.p.A.",
    "PRY.MI": "Prysmian S.p.A.",
    "REC.MI": "Recordati S.p.A.",
    "WBD.MI": "Webuild S.p.A.",
    "CPR.MI": "Campari S.p.A.",
    "FCT.MI": "Fincantieri S.p.A.",
    "PIRC.MI": "Pirelli S.p.A."
}

# ============================================================================
# CONFIGURAZIONE ANALISI
# ============================================================================

# Giornaliera
DAILY_PERIOD = "3mo"
DAILY_INTERVAL = "1d"
DAILY_MIN_POINTS = 30

# Settimanale
WEEKLY_PERIOD = "2y"
WEEKLY_INTERVAL = "1wk"
WEEKLY_MIN_POINTS = 30

# ============================================================================
# FUNZIONI UTILITY
# ============================================================================

def load_tickers(filename: str) -> list:
    """Carica lista ticker da file"""
    try:
        with open(filename, 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]
    except:
        return []

def load_config():
    """Carica configurazione personalizzata"""
    thresholds = DEFAULT_THRESHOLDS.copy()
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key in thresholds:
                            try:
                                thresholds[key] = float(value)
                            except:
                                pass
    except:
        pass
    
    return thresholds

def get_recommendation(score: float, thresholds: dict) -> tuple:
    """Restituisce raccomandazione basata su score"""
    if score < thresholds['STRONG_SELL']:
        return ("🔴🔴 VENDI SUBITO", "STRONG_SELL")
    elif score < thresholds['SELL']:
        return ("🔴 CONSIGLIA VENDITA", "SELL")
    elif score < thresholds['WARNING']:
        return ("🟡 MONITORA ATTIVAMENTE", "WARNING")
    elif score < thresholds['NEUTRAL']:
        return ("⚪ MANTIENI POSIZIONE", "NEUTRAL")
    elif score < thresholds['BUY']:
        return ("🟢 CONSIGLIA ACQUISTO", "BUY")
    else:
        return ("🟢🟢 FORTE ACQUISTO", "STRONG_BUY")

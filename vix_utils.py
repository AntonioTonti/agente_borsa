#!/usr/bin/env python3
"""
Utility per il calcolo del VIX Regime Score.

Combina due componenti:
- VIX Trend (60%): direzione del VIX negli ultimi giorni
- VIX Level (40%): livello assoluto con logica contrarian
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Dict


VIX_TICKER = "^VIX"


def fetch_vix_data(period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Scarica i dati del VIX. Ritorna DataFrame vuoto se fallisce."""
    try:
        df = yf.Ticker(VIX_TICKER).history(
            period=period, interval=interval, auto_adjust=False
        )
        if df.empty:
            return pd.DataFrame()
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        return df
    except Exception as e:
        print(f"   ⚠️ Errore download VIX: {e}")
        return pd.DataFrame()


def compute_vix_trend_score(vix_close: pd.Series) -> Tuple[float, str]:
    """
    Valuta la direzione del VIX: scende → bullish per equity; sale → bearish.
    Confronto VIX attuale vs media ultimi 5 giorni.
    """
    if len(vix_close) < 6:
        return 0.5, "N/D"

    vix_now = float(vix_close.iloc[-1])
    vix_ma5 = float(vix_close.tail(5).mean())

    if vix_ma5 <= 0:
        return 0.5, "N/D"

    pct_change = ((vix_now - vix_ma5) / vix_ma5) * 100.0

    if pct_change < -2.0:
        return 1.0, f"VIX in calo ({pct_change:+.1f}%) 🟢"
    elif pct_change > 2.0:
        return 0.2, f"VIX in salita ({pct_change:+.1f}%) 🔴"
    else:
        return 0.6, f"VIX stabile ({pct_change:+.1f}%) ⚪"


def compute_vix_level_score(vix_now: float) -> Tuple[float, str]:
    """
    Valuta il livello assoluto del VIX con logica contrarian.
    """
    if vix_now < 12:
        return 0.4, f"Compiacenza ({vix_now:.1f}) ⚠️"
    elif vix_now < 20:
        return 1.0, f"Normale ({vix_now:.1f}) 🟢"
    elif vix_now < 30:
        return 0.5, f"Caution ({vix_now:.1f}) ⚪"
    elif vix_now < 40:
        return 0.3, f"Stress ({vix_now:.1f}) 🔴"
    else:
        return 0.6, f"Panico ({vix_now:.1f}) ⚠️ possibile inversione"


def get_vix_regime() -> Dict:
    """
    Ritorna un dict con:
      - score: sub-score VIX finale (0..1)
      - vix_value: valore VIX corrente
      - message: descrizione testuale per Telegram
      - components: dizionario con le due sotto-componenti
    """
    default = {
        'score': 0.5,
        'vix_value': None,
        'message': "VIX: N/D",
        'components': {'trend': 0.5, 'level': 0.5},
    }

    df = fetch_vix_data()
    if df.empty:
        return default

    vix_close = df['Close']
    vix_now = float(vix_close.iloc[-1])

    trend_score, trend_msg = compute_vix_trend_score(vix_close)
    level_score, level_msg = compute_vix_level_score(vix_now)

    vix_score = (trend_score * 0.6) + (level_score * 0.4)

    message = f"VIX {vix_now:.1f} | {trend_msg} | {level_msg}"

    return {
        'score': round(vix_score, 3),
        'vix_value': round(vix_now, 2),
        'message': message,
        'components': {'trend': trend_score, 'level': level_score},
    }


if __name__ == "__main__":
    print("🧪 Test vix_utils.py")
    result = get_vix_regime()
    print(f"Score VIX: {result['score']}")
    print(f"Valore VIX: {result['vix_value']}")
    print(f"Messaggio: {result['message']}")

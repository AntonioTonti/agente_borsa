#!/usr/bin/env python3
"""
Modulo Risk Management per Agente Flash.
Calcola ATR, Stop-Loss, Take-Profit e Position Sizing raccomandato.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calcola l'Average True Range (ATR)."""
    if len(df) < period + 1:
        return 0.0
        
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr)

def compute_risk_levels(
    last_price: float, 
    atr: float, 
    direzione: str
) -> Tuple[float, float, str, float]:
    """
    Ritorna:
    (Stop_Loss, Take_Profit, Livello_Rischio, Position_Sizing_Pct)
    """
    if atr == 0.0 or last_price == 0.0:
        return 0.0, 0.0, "⚪ N/D", 0.0

    volatilità_pct = (atr / last_price) * 100.0
    
    # Classificazione Rischio
    if volatilità_pct < 1.2:
        risk_label = "🟢 Basso"
        sizing_pct = 15.0
    elif volatilità_pct < 2.5:
        risk_label = "🟡 Medio"
        sizing_pct = 10.0
    else:
        risk_label = "🔴 Alto"
        sizing_pct = 5.0

    # Stop Loss a 1.5 ATR e Take Profit a 3.0 ATR (Rapporto R/R = 1:2)
    if direzione == "BULLISH":
        stop_loss = last_price - (1.5 * atr)
        take_profit = last_price + (3.0 * atr)
    elif direzione == "BEARISH":
        stop_loss = last_price + (1.5 * atr)
        take_profit = last_price - (3.0 * atr)
    else:
        stop_loss = last_price - (1.5 * atr)
        take_profit = last_price + (1.5 * atr)

    return round(stop_loss, 2), round(take_profit, 2), risk_label, sizing_pct

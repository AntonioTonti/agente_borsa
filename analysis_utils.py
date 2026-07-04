#!/usr/bin/env python3
"""
Utility di analisi tecnica per gli agenti di trading
Contiene funzioni condivise per calcoli di trend, target e stop loss
"""

import pandas as pd
import numpy as np
from typing import Tuple

# ============================================================================
# FUNZIONE HEIKIN ASHI
# ============================================================================

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le barre Heikin Ashi a partire da un DataFrame OHLCV
    Restituisce un DataFrame con le colonne HA_Open, HA_High, HA_Low, HA_Close
    """
    ha = pd.DataFrame(index=df.index)
    
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_open = (df['Open'] + df['Close']) / 2
    
    ha['HA_Close'] = ha_close
    ha['HA_Open'] = ha_open.copy()
    
    for i in range(1, len(df)):
        ha.loc[df.index[i], 'HA_Open'] = (ha.loc[df.index[i-1], 'HA_Open'] + ha.loc[df.index[i-1], 'HA_Close']) / 2
    
    ha['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    for i in range(len(df)):
        ha.loc[df.index[i], 'HA_High'] = max(
            df.loc[df.index[i], 'High'],
            ha.loc[df.index[i], 'HA_Open'],
            ha.loc[df.index[i], 'HA_Close']
        )
        ha.loc[df.index[i], 'HA_Low'] = min(
            df.loc[df.index[i], 'Low'],
            ha.loc[df.index[i], 'HA_Open'],
            ha.loc[df.index[i], 'HA_Close']
        )
    
    return ha


# ============================================================================
# FUNZIONE PALLINO RIASSUNTIVO
# ============================================================================

def get_bullet(score: float) -> str:
    """
    Restituisce il pallino in base allo score finale
    🟢 Verde: score > 0.60 (trend rialzista)
    ⚪ Bianco: 0.40 <= score <= 0.60 (laterale/neutro)
    🔴 Rosso: score < 0.40 (trend ribassista)
    """
    if score > 0.60:
        return "🟢"
    elif score < 0.40:
        return "🔴"
    else:
        return "⚪"


# ============================================================================
# FUNZIONI DI STIMA TREND
# ============================================================================

def calculate_atr(close_prices: pd.Series, period: int = 14) -> float:
    """
    Calcola l'Average True Range (ATR) come misura di volatilità
    """
    if len(close_prices) < period + 1:
        return 0.0
    
    # Calcola True Range (semplificato usando i prezzi di chiusura)
    # In assenza di High/Low, usiamo Close come proxy
    high = close_prices
    low = close_prices
    prev_close = close_prices.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    
    return atr if not pd.isna(atr) else 0.0


def calculate_trend_estimate(close_prices: pd.Series, lookback: int = 7) -> Tuple[float, float, float]:
    """
    Calcola stima di trend, target e stop loss basati su regressione lineare e ATR
    
    Args:
        close_prices: Serie dei prezzi di chiusura
        lookback: Numero di barre da considerare (7 per daily, 3 per weekly)
    
    Returns:
        tuple: (variazione_percentuale, target_price, stop_loss)
    """
    if len(close_prices) < lookback + 2:
        return 0.0, close_prices.iloc[-1], close_prices.iloc[-1]
    
    # Prendi gli ultimi N prezzi
    prices = close_prices.iloc[-lookback:].values
    
    # Calcola ATR (volatilità)
    atr = calculate_atr(close_prices, 14)
    
    # Se ATR è zero, usa un valore minimo
    if atr == 0 or pd.isna(atr):
        atr = close_prices.std() * 0.5
        if atr == 0:
            atr = 0.01
    
    # Regressione lineare
    x = np.arange(len(prices))
    
    # Calcolo manuale della regressione lineare (evita dipendenza da scipy)
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(prices)
    sum_xy = np.sum(x * prices)
    sum_xx = np.sum(x * x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Variazione percentuale stimata
    last_price = prices[-1]
    if last_price > 0:
        var_percent = (slope * lookback / last_price) * 100
    else:
        var_percent = 0.0
    
    # Limita la variazione percentuale a un range ragionevole (±30%)
    var_percent = max(-30.0, min(30.0, var_percent))
    
    # Target basato su regressione + volatilità
    projected_price = intercept + slope * (len(prices) + lookback)
    
    if var_percent > 3:  # Trend positivo
        target_price = projected_price + (atr * 0.8)
        stop_loss = last_price - (atr * 1.8)
    elif var_percent < -3:  # Trend negativo
        target_price = projected_price - (atr * 0.8)
        stop_loss = last_price + (atr * 1.8)
    else:  # Trend laterale
        target_price = projected_price
        stop_loss = last_price - (atr * 1.0)
    
    # Arrotonda a 2 decimali
    target_price = round(float(target_price), 2)
    stop_loss = round(float(stop_loss), 2)
    var_percent = round(float(var_percent), 1)
    
    # Assicura che stop_loss sia diverso dal target
    if abs(stop_loss - target_price) < 0.01:
        if var_percent > 0:
            stop_loss = round(stop_loss - atr * 0.5, 2)
        else:
            stop_loss = round(stop_loss + atr * 0.5, 2)
    
    return var_percent, target_price, stop_loss


def format_trend_line(var_percent: float, target_price: float, stop_loss: float) -> str:
    """
    Formatta la linea di trend per il report Telegram
    """
    if var_percent > 2:
        return f"   📈 Trend: +{var_percent}% | Target: {target_price} | Stop Loss: {stop_loss}"
    elif var_percent < -2:
        return f"   📉 Trend: {var_percent}% | Target: {target_price} | Stop Loss: {stop_loss}"
    else:
        return f"   ➡️ Trend laterale | Target: {target_price} | Stop Loss: {stop_loss}"

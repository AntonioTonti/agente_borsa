import numpy as np
import pandas as pd
import ta


def get_bullet(score: float) -> str:
    """Ritorna l'icona/bullet appropriata in base al valore dello score."""
    if score >= 0.70:
        return "🟢"
    elif score >= 0.50:
        return "🟡"
    elif score >= 0.35:
        return "🟠"
    else:
        return "🔴"


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola le candele Heikin Ashi."""
    ha_df = df.copy()
    ha_df['ha_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    ha_open = np.zeros(len(df))
    if len(df) > 0:
        ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + ha_df['ha_close'].iloc[i-1]) / 2
            
    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = ha_df[['High', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['Low', 'ha_open', 'ha_close']].min(axis=1)
    
    return ha_df


def analyze_ticker(df: pd.DataFrame, ticker: str, description: str = "") -> dict:
    """
    Esegue l'analisi tecnica completa calcolando indicatori, score e 
    generando la lista estesa di tutti i segnali rilevati per report e grafici HTML.
    """
    if df.empty or len(df) < 35:
        return None

    # Calcolo Medie e Indicatori Base
    df = calculate_heikin_ashi(df)
    df['ema10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['ma31'] = ta.trend.sma_indicator(df['Close'], window=31)
    
    # RSI
    rsi_series = ta.momentum.rsi(df['Close'], window=14)
    df['rsi'] = rsi_series
    
    # MACD
    macd_obj = ta.trend.MACD(df['Close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    
    # Dati Ultima Candela
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    close_val = last['Close']
    prev_close = prev['Close']
    close_change_pct = ((close_val - prev_close) / prev_close) * 100
    
    ema10_val = last['ema10']
    ma31_val = last['ma31']
    rsi_val = last['rsi']
    macd_val = last['macd']
    macd_sig_val = last['macd_signal']
    
    # Delta EMA/MA
    delta_ema_ma_pct = ((ema10_val - ma31_val) / ma31_val) * 100
    df['delta_abs'] = ((df['ema10'] - df['ma31']).abs() / df['ma31']) * 100
    avg_delta_3m = df['delta_abs'].tail(63).mean()
    
    # Volumi
    vol_val = last['Volume']
    vol_avg_3m = df['Volume'].tail(63).mean()
    vol_diff_pct = ((vol_val - vol_avg_3m) / vol_avg_3m) * 100 if vol_avg_3m > 0 else 0
    
    # Heikin Ashi Body & Shadows
    ha_open = last['ha_open']
    ha_close = last['ha_close']
    ha_high = last['ha_high']
    ha_low = last['ha_low']
    
    ha_body = abs(ha_close - ha_open)
    df['ha_body_val'] = abs(df['ha_close'] - df['ha_open'])
    avg_ha_body_3m = df['ha_body_val'].tail(63).mean()
    body_ratio = ha_body / avg_ha_body_3m if avg_ha_body_3m > 0 else 1.0
    
    is_ha_green = ha_close >= ha_open
    upper_shadow = ha_high - max(ha_open, ha_close)
    lower_shadow = min(ha_open, ha_close) - ha_low
    
    # ZigZag Semplificato
    zigzag_status = "Rialzista" if close_val >= ema10_val else "Ribassista"
    
    # Target e Stop Loss
    atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
    target_price = close_val + (1.5 * atr)
    stop_loss_price = close_val - (1.0 * atr)
    
    # Calcolo Score (0.0 - 1.0)
    score_components = []
    
    # 1. EMA vs MA (30%)
    if ema10_val > ma31_val:
        score_components.append(0.30)
    
    # 2. Heikin Ashi Verde (25%)
    if is_ha_green:
        score_components.append(0.25)
        
    # 3. MACD > Signal (20%)
    if macd_val > macd_sig_val:
        score_components.append(0.20)
        
    # 4. RSI (15%)
    if 45 <= rsi_val <= 65:
        score_components.append(0.15)
    elif 35 <= rsi_val < 45 or 65 < rsi_val <= 70:
        score_components.append(0.08)
        
    # 5. Volumi sopra media (10%)
    if vol_diff_pct > 0:
        score_components.append(0.10)
        
    final_score = round(sum(score_components), 3)

    # --- COSTRUZIONE LISTA COMPLETA SEGNALI ---
    signals_list = []
    
    # 1. EMA / MA
    ema_icon = "🟢" if ema10_val > ma31_val else "🔴"
    signals_list.append(f"{ema_icon} EMA10 ({ema10_val:.2f}) {'sopra' if ema10_val > ma31_val else 'sotto'} MA31 ({ma31_val:.2f})")
    signals_list.append(f"📐 Delta EMA10/MA31: {delta_ema_ma_pct:+.2f}% (Media Abs 3M: {avg_delta_3m:.2f}%)")
    
    # 2. Heikin Ashi Body & State
    body_desc = "Corpo esteso" if body_ratio > 1.2 else ("Corpo contenuto" if body_ratio < 0.8 else "Corpo nella media")
    signals_list.append(f"🕯️ Espansione Corpo HA: {body_ratio:.1f}x media 3M ({body_desc})")
    
    ha_color_icon = "🟢" if is_ha_green else "🔴"
    shadow_desc = "OMBRA SUP. PREVALENTE" if upper_shadow > lower_shadow else "OMBRA INF. PREVALENTE"
    color_text = "VERDE" if is_ha_green else "ROSSA"
    signals_list.append(f"{ha_color_icon} HEIKIN ASHI: {color_text} CON {shadow_desc}")
    
    # 3. ZigZag
    zz_icon = "⚡"
    signals_list.append(f"{zz_icon} ZIGZAG: {zigzag_status}")
    
    # 4. Volumi
    vol_icon = "📊"
    vol_text = "sopra" if vol_diff_pct >= 0 else "sotto"
    signals_list.append(f"{vol_icon} Volume: {vol_text} media 3 mesi ({vol_diff_pct:+.0f}%)")
    
    # 5. Chiusura vs Precedente
    change_icon = "🔷" if close_change_pct >= 0 else "🔻"
    signals_list.append(f"{change_icon} Chiusura vs Prec: {close_change_pct:+.2f}%")
    
    # 6. RSI
    rsi_state = "Ipercomprato" if rsi_val > 70 else ("Ipervenduto" if rsi_val < 30 else "Neutro")
    signals_list.append(f"📊 RSI: {rsi_val:.1f} ({rsi_state})")
    
    # 7. MACD
    macd_icon = "🟢" if macd_val > macd_sig_val else "🔴"
    macd_rel = "sopra" if macd_val > macd_sig_val else "sotto"
    signals_list.append(f"{macd_icon} MACD ({macd_val:.4f}) {macd_rel} Signal ({macd_sig_val:.4f})")
    
    # 8. Trend & Livelli
    trend_icon = "📈" if final_score >= 0.6 else ("➡️" if final_score >= 0.4 else "📉")
    trend_label = "Rialzista" if final_score >= 0.6 else ("Laterale" if final_score >= 0.4 else "Ribassista")
    signals_list.append(f"{trend_icon} Trend {trend_label} | Target: {target_price:.2f} | Stop Loss: {stop_loss_price:.2f}")

    return {
        'ticker': ticker,
        'description': description,
        'score': final_score,
        'close': close_val,
        'change_pct': close_change_pct,
        'signals': signals_list,
        'data': df
    }


def generate_signals_html(signals: list) -> str:
    """Formatta la lista dei segnali in elementi HTML <li> per la pagina web."""
    if not signals:
        return "<li>Nessun segnale rilevato</li>"
    return "\n".join([f"<li>{sig}</li>" for sig in signals])

import os
import sys
from datetime import datetime, timedelta
import requests
import yfinance as yf
import ta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Dizionario dei ticker con descrizioni
TICKER_DESCRIPTIONS = {
    "STM": "STMicroelectronics N.V.",
    "SPM.MI": "Saipem",
    "AMP.MI": "Amplifon",
    "ZV.MI": "Zignago Vetro",
    "NEXI.MI": "Nexi",
    "TIT.MI": "Telecom Italia",
    "BSS.MI": "Biesse",
    "TSL.MI": "Tessellis",
    "PRY.MI": "Prysmian",
    "REC.MI": "Recordati",
    "WBD.MI": "Webuild",
    "CPR.MI": "Campari",
    "FCT.MI": "Fincantieri",
    "PIRC.MI": "Pirelli",
    "RACE.MI": "Ferrari"
}

# File di configurazione
PORTFOLIO_FILE = "portfolio.txt"
MIN_DATA_POINTS = 100  # Punti dati minimi per l'analisi (1 anno)
ZIGZAG_DEPTH = 10      # Profondità analisi ZigZag
VOLUME_MA_DAYS = 20    # Giorni per media volume
RSI_PERIOD = 14        # Periodo RSI
BB_PERIOD = 20         # Periodo Bande di Bollinger
BB_STD = 2             # Deviazioni standard Bollinger

# Soglie
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_THRESHOLD = 1.5  # Volume 50% sopra la media

# ============================================================================
# FUNZIONI DI SUPPORTO
# ============================================================================

def load_portfolio_with_sections(filename: str = PORTFOLIO_FILE) -> Dict[str, List[str]]:
    """
    Carica la lista dei ticker dal file portfolio con sezioni.
    
    Ritorna: {"PORTAFOGLIO": [tickers], "OSSERVATI": [tickers]}
    """
    sections = {
        "PORTAFOGLIO": [],
        "OSSERVATI": []
    }
    
    try:
        with open(filename, 'r') as f:
            current_section = None
            
            for line in f:
                line = line.strip().upper()
                
                if not line:
                    continue
                    
                if line == "PORTAFOGLIO":
                    current_section = "PORTAFOGLIO"
                elif line == "OSSERVATI":
                    current_section = "OSSERVATI"
                elif current_section and line not in ["PORTAFOGLIO", "OSSERVATI"]:
                    sections[current_section].append(line)
        
        print(f"📋 Tickers caricati: {len(sections['PORTAFOGLIO'])} in Portafoglio, {len(sections['OSSERVATI'])} Osservati")
        return sections
        
    except FileNotFoundError:
        print(f"❌ File {filename} non trovato")
        return sections
    except Exception as e:
        print(f"❌ Errore nel caricamento del portfolio: {e}")
        return sections


def download_ticker_data(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Scarica i dati del ticker da Yahoo Finance (1 anno).
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            print(f"⚠️  {ticker}: Nessun dato disponibile")
            return None
        
        # Gestione MultiIndex di yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        return df
    except Exception as e:
        print(f"❌ {ticker}: Errore nel download dati - {e}")
        return None


def calculate_heikin_ashi_color(df: pd.DataFrame) -> str:
    """
    Calcola il colore della candela Heikin Ashi.
    """
    try:
        if len(df) < 2:
            return "UNKNOWN"
            
        ha_close = (df['Open'].iloc[-1] + df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 4
        ha_open = (df['Open'].iloc[-2] + df['Close'].iloc[-2]) / 2
        
        return "BULL" if ha_close > ha_open else "BEAR"
    except Exception as e:
        print(f"⚠️  Errore nel calcolo Heikin Ashi: {e}")
        return "UNKNOWN"


def analyze_zigzag_direction(df: pd.DataFrame, depth: int = ZIGZAG_DEPTH) -> str:
    """
    Analizza la direzione ZigZag basata sui massimi/minimi recenti.
    """
    try:
        close_prices = df['Close']
        
        if len(close_prices) < depth:
            depth = len(close_prices)
            
        recent_prices = close_prices.tail(depth)
        current_price = float(close_prices.iloc[-1])
        
        recent_max = float(recent_prices.max())
        recent_min = float(recent_prices.min())
        
        up_threshold = recent_max * 0.98
        down_threshold = recent_min * 1.02
        
        if current_price > up_threshold:
            return "UP"
        elif current_price < down_threshold:
            return "DOWN"
        else:
            return "FLAT"
    except Exception as e:
        print(f"⚠️  Errore nell'analisi ZigZag: {e}")
        return "FLAT"


def detect_ma_ema_crossover(df: pd.DataFrame) -> Optional[str]:
    """
    Rileva incroci tra MA31 ed EMA10.
    """
    try:
        close_prices = df['Close']
        
        if len(close_prices) < 32:
            return None
        
        ma31 = ta.trend.sma_indicator(close_prices, window=31)
        ema10 = ta.trend.ema_indicator(close_prices, window=10)
        
        if len(ma31) < 2 or len(ema10) < 2:
            return None
        
        ma31_current = float(ma31.iloc[-1])
        ema10_current = float(ema10.iloc[-1])
        ma31_prev = float(ma31.iloc[-2])
        ema10_prev = float(ema10.iloc[-2])
        
        # Crossover UP: EMA passa sopra MA
        if ema10_prev <= ma31_prev and ema10_current > ma31_current:
            return "UP"
        
        # Crossover DOWN: MA passa sopra EMA
        elif ma31_prev <= ema10_prev and ma31_current > ema10_current:
            return "DOWN"
        
        return "NO_CROSS"
        
    except Exception as e:
        print(f"⚠️  Errore nel rilevamento crossover MA/EMA: {e}")
        return None


def get_ma_ema_position(df: pd.DataFrame) -> Optional[str]:
    """
    Ottiene la posizione attuale tra MA31 e EMA10.
    """
    try:
        close_prices = df['Close']
        
        if len(close_prices) < 32:
            return None
        
        ma31 = ta.trend.sma_indicator(close_prices, window=31)
        ema10 = ta.trend.ema_indicator(close_prices, window=10)
        
        if len(ma31) == 0 or len(ema10) == 0:
            return None
        
        ma31_current = float(ma31.iloc[-1])
        ema10_current = float(ema10.iloc[-1])
        
        if ema10_current > ma31_current:
            return "UP"
        else:
            return "DOWN"
            
    except Exception as e:
        print(f"⚠️  Errore nel calcolo posizione MA/EMA: {e}")
        return None


def calculate_composite_index(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Calcola l'INDICE COMPOSITO basato su 6 indicatori.
    
    Ritorna: (🟢 INDICE UP / 🔴 INDICE DOWN, [dettagli indicatori])
    """
    indicators = []
    score_up = 0
    total_indicators = 6  # RSI, MACD, Bollinger, Volume, Supporti/Resistenze, Trend
    
    try:
        close_prices = df['Close']
        volume = df['Volume']
        
        # 1. RSI (14 periodi)
        if len(close_prices) >= RSI_PERIOD:
            rsi = ta.momentum.RSIIndicator(close_prices, window=RSI_PERIOD).rsi()
            rsi_current = float(rsi.iloc[-1])
            
            if rsi_current < RSI_OVERSOLD:
                score_up += 1
                indicators.append(f"RSI {rsi_current:.1f} (Oversold)")
            elif rsi_current > RSI_OVERBOUGHT:
                indicators.append(f"RSI {rsi_current:.1f} (Overbought)")
            else:
                score_up += 0.5  # Neutrale
                indicators.append(f"RSI {rsi_current:.1f}")
        
        # 2. MACD
        if len(close_prices) >= 26:
            macd_ind = ta.trend.MACD(close_prices)
            macd_line = macd_ind.macd()
            signal_line = macd_ind.macd_signal()
            
            macd_current = float(macd_line.iloc[-1])
            signal_current = float(signal_line.iloc[-1])
            
            if macd_current > signal_current:
                score_up += 1
                indicators.append("MACD ↑")
            else:
                indicators.append("MACD ↓")
        
        # 3. Bande di Bollinger
        if len(close_prices) >= BB_PERIOD:
            bb = ta.volatility.BollingerBands(close_prices, window=BB_PERIOD, window_dev=BB_STD)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            
            current_price = float(close_prices.iloc[-1])
            bb_upper_current = float(bb_upper.iloc[-1])
            bb_lower_current = float(bb_lower.iloc[-1])
            
            if current_price > bb_upper_current:
                indicators.append(f"Bollinger: {current_price:.2f} > Upper {bb_upper_current:.2f}")
            elif current_price < bb_lower_current:
                score_up += 1  # Prezzo vicino al supporto
                indicators.append(f"Bollinger: {current_price:.2f} < Lower {bb_lower_current:.2f}")
            else:
                score_up += 0.5
                indicators.append(f"Bollinger: Middle band")
        
        # 4. Analisi Volume
        if len(volume) >= VOLUME_MA_DAYS:
            volume_ma = volume.rolling(window=VOLUME_MA_DAYS).mean()
            current_volume = float(volume.iloc[-1])
            volume_ma_current = float(volume_ma.iloc[-1])
            
            if volume_ma_current > 0 and current_volume > volume_ma_current * VOLUME_THRESHOLD:
                score_up += 1
                indicators.append(f"Volume ↑ {current_volume/1e6:.1f}M > MA {volume_ma_current/1e6:.1f}M")
            else:
                indicators.append(f"Volume {current_volume/1e6:.1f}M")
        
        # 5. Supporti/Resistenze (basato su medie mobili)
        if len(close_prices) >= 50:
            # SMA50 come supporto/resistenza
            sma50 = ta.trend.sma_indicator(close_prices, window=50)
            sma50_current = float(sma50.iloc[-1])
            current_price = float(close_prices.iloc[-1])
            
            if current_price > sma50_current:
                score_up += 1
                indicators.append(f"SMA50 Support: {current_price:.2f} > {sma50_current:.2f}")
            else:
                indicators.append(f"SMA50 Resistance: {current_price:.2f} < {sma50_current:.2f}")
        
        # 6. Trend Generale (SMA200)
        if len(close_prices) >= 200:
            sma200 = ta.trend.sma_indicator(close_prices, window=200)
            sma200_current = float(sma200.iloc[-1])
            current_price = float(close_prices.iloc[-1])
            
            if current_price > sma200_current:
                score_up += 1
                indicators.append(f"SMA200 ↑: {current_price:.2f} > {sma200_current:.2f}")
            else:
                indicators.append(f"SMA200 ↓: {current_price:.2f} < {sma200_current:.2f}")
        
        # Calcola percentuale e determina INDICE
        if total_indicators > 0:
            score_percentage = (score_up / total_indicators) * 100
            
            if score_percentage >= 60:
                return "🟢 INDICE UP", indicators
            elif score_percentage <= 40:
                return "🔴 INDICE DOWN", indicators
            else:
                return "⚪ INDICE NEUTRO", indicators
        else:
            return "⚪ INDICE NON CALCOLATO", []
            
    except Exception as e:
        print(f"⚠️  Errore nel calcolo indice composito: {e}")
        return "⚪ INDICE ERRORE", []


def analyze_ticker_signals(ticker: str) -> Dict[str, List[str]]:
    """
    Analizza un singolo ticker e restituisce tutti i segnali.
    
    Ritorna: {
        "ma_ema": [segnali MA/EMA],
        "zigzag": [segnali ZigZag],
        "heikin": [segnali Heikin Ashi],
        "indice": (indice, [dettagli]),
        "description": descrizione_ticker
    }
    """
    signals = {
        "ma_ema": [],
        "zigzag": [],
        "heikin": [],
        "indice": ("", []),
        "description": TICKER_DESCRIPTIONS.get(ticker, ticker)
    }
    
    # Scarica dati (1 anno)
    df = download_ticker_data(ticker, period="1y")
    if df is None or len(df) < MIN_DATA_POINTS:
        return signals
    
    try:
        # 1. Analisi MA/EMA
        crossover_signal = detect_ma_ema_crossover(df)
        ma_ema_position = get_ma_ema_position(df)
        
        if crossover_signal == "UP":
            signals["ma_ema"].append("📈 CROSSOVER UP")
        elif crossover_signal == "DOWN":
            signals["ma_ema"].append("📉 CROSSOVER DOWN")
        
        if ma_ema_position == "UP":
            signals["ma_ema"].append("🟢 EMA10 > MA31")
        elif ma_ema_position == "DOWN":
            signals["ma_ema"].append("🔴 MA31 > EMA10")
        
        # 2. Analisi ZigZag
        zz_dir = analyze_zigzag_direction(df)
        if zz_dir != "FLAT":
            signals["zigzag"].append(f"📊 ZigZag: {zz_dir}")
        
        # 3. Analisi Heikin Ashi
        if len(df) > 2:
            color_current = calculate_heikin_ashi_color(df.iloc[-2:])
            color_previous = calculate_heikin_ashi_color(df.iloc[-3:-1])
            
            if color_current != color_previous and color_current != "UNKNOWN":
                signals["heikin"].append(f"🕯️ Heikin Ashi: {color_current}")
        
        # 4. Indice Composito
        index_result, index_details = calculate_composite_index(df)
        signals["indice"] = (index_result, index_details)
        
        return signals
        
    except Exception as e:
        print(f"❌ {ticker}: Errore nell'analisi - {e}")
        return signals


# ============================================================================
# FORMATTAZIONE TELEGRAM
# ============================================================================

def format_ticker_signals_for_telegram(ticker: str, signals: Dict[str, List[str]]) -> str:
    """
    Formatta i segnali di un singolo ticker per Telegram.
    """
    parts = []
    
    # Solo descrizione (senza ticker e senza link)
    description = signals.get("description", ticker)
    ticker_line = f"*{description}*"
    parts.append(ticker_line)
    
    # Indice Composito (prima cosa)
    index_result, index_details = signals["indice"]
    if index_result:
        parts.append(index_result)
    
    # Tutti gli altri segnali
    all_signals = []
    all_signals.extend(signals.get("ma_ema", []))
    all_signals.extend(signals.get("zigzag", []))
    all_signals.extend(signals.get("heikin", []))
    
    if all_signals:
        parts.append("\n".join(all_signals))
    
    return "\n".join(parts)


def format_all_signals_for_telegram(portfolio_signals: Dict[str, Dict], osservati_signals: Dict[str, Dict]) -> str:
    """
    Formatta tutti i segnali per l'invio su Telegram.
    """
    current_time = datetime.now().strftime("%d/%m %H:%M")
    header = f"📈 *SEGNALI BORSA {current_time}*\n\n"
    
    message_parts = [header]
    
    # Sezione PORTAFOGLIO
    if portfolio_signals:
        message_parts.append("════════════════════════")
        message_parts.append("*📊 PORTAFOGLIO*")
        message_parts.append("════════════════════════")
        
        for ticker, signals in portfolio_signals.items():
            formatted = format_ticker_signals_for_telegram(ticker, signals)
            message_parts.append(formatted)
            message_parts.append("─────────────────────")
    
    # Sezione OSSERVATI
    if osservati_signals:
        message_parts.append("════════════════════════")
        message_parts.append("*👀 OSSERVATI*")
        message_parts.append("════════════════════════")
        
        for ticker, signals in osservati_signals.items():
            formatted = format_ticker_signals_for_telegram(ticker, signals)
            message_parts.append(formatted)
            message_parts.append("─────────────────────")
    
    # Rimuovi l'ultima linea se è "─────────────────────"
    if message_parts[-1] == "─────────────────────":
        message_parts.pop()
    
    return "\n".join(message_parts)


def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    """
    Invia un messaggio a Telegram.
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True  # Disabilitato visto che non ci sono link
        }
        
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Errore Telegram: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Timeout nella connessione a Telegram")
        return False
    except Exception as e:
        print(f"❌ Errore nell'invio a Telegram: {e}")
        return False


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """
    Funzione principale dell'agente di trading.
    """
    print(f"🤖 AVVIO AGENTE BORSA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Scaricando dati 1 anno per analisi completa")
    
    # Carica credenziali Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not telegram_token or not telegram_chat_id:
        print("⚠️  Credenziali Telegram non configurate")
        print("   Assicurati di avere TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID")
    
    # Carica portfolio con sezioni
    sections = load_portfolio_with_sections()
    
    if not sections["PORTAFOGLIO"] and not sections["OSSERVATI"]:
        print("❌ Nessun ticker trovato nel portfolio")
        sys.exit(1)
    
    # Analizza tutti i ticker
    portfolio_signals = {}
    osservati_signals = {}
    
    print(f"\n🔍 Analisi in corso...")
    
    # Analizza PORTAFOGLIO
    if sections["PORTAFOGLIO"]:
        print(f"\n📊 Analizzando {len(sections['PORTAFOGLIO'])} titoli in PORTAFOGLIO:")
        for ticker in sections["PORTAFOGLIO"]:
            signals = analyze_ticker_signals(ticker)
            if any(signals.values()):  # Solo se ci sono segnali
                portfolio_signals[ticker] = signals
                # Log nel terminale
                time_str = datetime.now().strftime('%H:%M')
                description = signals.get("description", ticker)
                ma_ema_str = " | ".join(signals.get("ma_ema", []))
                if ma_ema_str:
                    print(f"{time_str} 📬 {description}: {ma_ema_str}")
    
    # Analizza OSSERVATI
    if sections["OSSERVATI"]:
        print(f"\n👀 Analizzando {len(sections['OSSERVATI'])} titoli OSSERVATI:")
        for ticker in sections["OSSERVATI"]:
            signals = analyze_ticker_signals(ticker)
            if any(signals.values()):  # Solo se ci sono segnali
                osservati_signals[ticker] = signals
                # Log nel terminale
                time_str = datetime.now().strftime('%H:%M')
                description = signals.get("description", ticker)
                ma_ema_str = " | ".join(signals.get("ma_ema", []))
                if ma_ema_str:
                    print(f"{time_str} 👁️ {description}: {ma_ema_str}")
    
    # Riepilogo
    print(f"\n📈 RIEPILOGO:")
    print(f"   Titoli Portafoglio: {len(sections['PORTAFOGLIO'])}")
    print(f"   Titoli Osservati: {len(sections['OSSERVATI'])}")
    print(f"   Segnali Portafoglio: {len(portfolio_signals)}")
    print(f"   Segnali Osservati: {len(osservati_signals)}")
    
    # Invia a Telegram se ci sono segnali
    if (portfolio_signals or osservati_signals) and telegram_token and telegram_chat_id:
        print("\n📤 Invio segnali a Telegram...")
        
        message = format_all_signals_for_telegram(portfolio_signals, osservati_signals)
        success = send_telegram_message(telegram_token, telegram_chat_id, message)
        
        if success:
            total_signals = len(portfolio_signals) + len(osservati_signals)
            print(f"✅ Messaggio inviato a Telegram: {total_signals} titoli con segnali")
        else:
            print("❌ Invio a Telegram fallito")
    elif portfolio_signals or osservati_signals:
        print("ℹ️  Segnali rilevati ma non inviati (credenziali Telegram mancanti)")
    else:
        print("ℹ️  Nessun segnale da inviare")
    
    print(f"\n🏁 ELABORAZIONE COMPLETATA - {datetime.now().strftime('%H:%M:%S')}")


# ============================================================================
# PUNTO DI INGRESSO
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interruzione manuale dell'agente")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


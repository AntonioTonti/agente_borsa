import os
import sys
from datetime import datetime
import requests
import yfinance as yf
import ta
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Dizionario dei ticker con descrizioni
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

# File di configurazione
PORTFOLIO_FILE = "portfolio.txt"
MIN_DATA_POINTS = 50  # Punti dati minimi per l'analisi
ZIGZAG_DEPTH = 10     # Profondità analisi ZigZag

# ============================================================================
# FUNZIONI DI SUPPORTO
# ============================================================================

def load_portfolio(filename: str = PORTFOLIO_FILE) -> List[str]:
    """
    Carica la lista dei ticker dal file portfolio.
    
    Args:
        filename: Nome del file contenente i ticker
        
    Returns:
        Lista di ticker in maiuscolo
    """
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        
        if not tickers:
            print("⚠️  Nessun ticker trovato nel portfolio")
            
        return tickers
    except FileNotFoundError:
        print(f"❌ File {filename} non trovato")
        return []
    except Exception as e:
        print(f"❌ Errore nel caricamento del portfolio: {e}")
        return []


def download_ticker_data(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Scarica i dati del ticker da Yahoo Finance.
    
    Args:
        ticker: Simbolo del titolo
        period: Periodo storico (es. "6mo", "1y")
        interval: Intervallo temporale (es. "1d", "1h")
        
    Returns:
        DataFrame con i dati o None in caso di errore
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            print(f"⚠️  {ticker}: Nessun dato disponibile")
            return None
            
        # Verifica che abbiamo tutte le colonne necessarie
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ {ticker}: Colonna {col} mancante")
                return None
                
        return df
    except Exception as e:
        print(f"❌ {ticker}: Errore nel download dati - {e}")
        return None


def calculate_heikin_ashi_color(df: pd.DataFrame) -> str:
    """
    Calcola il colore della candela Heikin Ashi.
    
    Args:
        df: DataFrame con dati OHLC
        
    Returns:
        "BULL" se bullish, "BEAR" se bearish
    """
    try:
        # Calcola Heikin Ashi Close
        ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Calcola Heikin Ashi Open
        ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
        
        # Ultimi valori
        last_ha_close = ha_close.iloc[-1]
        last_ha_open = ha_open.iloc[-1]
        
        return "BULL" if last_ha_close > last_ha_open else "BEAR"
    except Exception as e:
        print(f"⚠️  Errore nel calcolo Heikin Ashi: {e}")
        return "UNKNOWN"


def analyze_zigzag_direction(df: pd.DataFrame, depth: int = ZIGZAG_DEPTH) -> str:
    """
    Analizza la direzione ZigZag basata sui massimi/minimi recenti.
    
    Args:
        df: DataFrame con dati OHLC
        depth: Numero di giorni da analizzare
        
    Returns:
        "UP", "DOWN" o "FLAT"
    """
    try:
        close_prices = df['Close']
        
        if len(close_prices) < depth:
            depth = len(close_prices)
            
        recent_prices = close_prices.tail(depth)
        current_price = close_prices.iloc[-1]
        
        # Calcola massimo e minimo del periodo recente
        recent_max = recent_prices.max()
        recent_min = recent_prices.min()
        
        # Soglie per determinare la direzione
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


def detect_ma_crossover(df: pd.DataFrame) -> Optional[int]:
    """
    Rileva incroci tra MA31 ed EMA10.
    
    Args:
        df: DataFrame con dati OHLC
        
    Returns:
        1 per crossover rialzista, -1 per ribassista, 0 per nessun cambio, None per errore
    """
    try:
        close_prices = df['Close']
        
        # Calcola medie mobili
        ma31 = ta.trend.sma_indicator(close_prices, window=31)
        ema10 = ta.trend.ema_indicator(close_prices, window=10)
        
        # Identifica crossover
        position = np.where(ma31 > ema10, 1, 0)
        crossover_signal = pd.Series(position).diff().iloc[-1]
        
        return int(crossover_signal) if not pd.isna(crossover_signal) else 0
    except Exception as e:
        print(f"⚠️  Errore nel rilevamento crossover MA: {e}")
        return None


# ============================================================================
# ANALISI SEGNALI
# ============================================================================

def analyze_ticker_signals(ticker: str) -> List[str]:
    """
    Analizza un singolo ticker e restituisce i segnali rilevati.
    
    Args:
        ticker: Simbolo del titolo
        
    Returns:
        Lista di segnali rilevati (descrizioni)
    """
    signals = []
    
    # Scarica dati
    df = download_ticker_data(ticker)
    if df is None or len(df) < MIN_DATA_POINTS:
        return signals
    
    try:
        # 1. Analisi crossover MA31/EMA10
        ma_signal = detect_ma_crossover(df)
        if ma_signal == 1:
            signals.append("🟢 Incrocio rialzista MA31/EMA10")
        elif ma_signal == -1:
            signals.append("🔴 Incrocio ribassista MA31/EMA10")
        
        # 2. Analisi direzione ZigZag
        zz_dir = analyze_zigzag_direction(df)
        if zz_dir != "FLAT":
            signals.append(f"📊 ZigZag: {zz_dir}")
        
        # 3. Analisi Heikin Ashi
        # Calcola colore attuale e precedente
        if len(df) > 1:
            color_current = calculate_heikin_ashi_color(df.iloc[-1:])
            color_previous = calculate_heikin_ashi_color(df.iloc[-2:-1])
            
            if color_current != color_previous:
                signals.append(f"🕯️ Heikin Ashi: {color_current}")
        
        return signals
        
    except Exception as e:
        print(f"❌ {ticker}: Errore nell'analisi - {e}")
        return []


# ============================================================================
# COMUNICAZIONE TELEGRAM
# ============================================================================

def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    """
    Invia un messaggio a Telegram.
    
    Args:
        token: Token del bot Telegram
        chat_id: ID della chat
        message: Messaggio da inviare
        
    Returns:
        True se inviato con successo, False altrimenti
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
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


def format_signals_for_telegram(ticker_signals: Dict[str, List[str]]) -> str:
    """
    Formatta i segnali per l'invio su Telegram.
    
    Args:
        ticker_signals: Dizionario {ticker: [segnali]}
        
    Returns:
        Messaggio formattato per Telegram
    """
    if not ticker_signals:
        return "📭 Nessun segnale rilevato"
    
    current_time = datetime.now().strftime("%d/%m %H:%M")
    header = f"📈 *Segnali Borsa {current_time}*\n\n"
    
    message_parts = []
    
    for ticker, signals in ticker_signals.items():
        if signals:
            description = TICKER_DESCRIPTIONS.get(ticker, ticker)
            ticker_header = f"*{ticker}* – {description}"
            signals_text = "\n".join(signals)
            message_parts.append(f"{ticker_header}\n{signals_text}")
    
    return header + "\n\n".join(message_parts)


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """
    Funzione principale dell'agente di trading.
    """
    print(f"🤖 Avvio Agente Borsa - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Carica credenziali Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not telegram_token or not telegram_chat_id:
        print("⚠️  Credenziali Telegram non configurate")
        print("   Assicurati di avere TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID")
    
    # Carica portfolio
    portfolio = load_portfolio()
    if not portfolio:
        print("❌ Portfolio vuoto o non trovato")
        sys.exit(1)
    
    print(f"📊 Analizzando {len(portfolio)} titoli...")
    
    # Analizza ogni ticker
    all_signals = {}
    signals_found = 0
    
    for ticker in portfolio:
        signals = analyze_ticker_signals(ticker)
        
        if signals:
            all_signals[ticker] = signals
            signals_found += len(signals)
            
            # Log nel terminale
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
            print(f"{time_str} 📬 {ticker}: {' | '.join(signals)}")
    
    # Riepilogo
    print(f"\n📊 Riepilogo:")
    print(f"   Titoli analizzati: {len(portfolio)}")
    print(f"   Titoli con segnali: {len(all_signals)}")
    print(f"   Segnali totali: {signals_found}")
    
    # Invia a Telegram se ci sono segnali e le credenziali sono valide
    if all_signals and telegram_token and telegram_chat_id:
        print("\n📤 Invio segnali a Telegram...")
        
        message = format_signals_for_telegram(all_signals)
        success = send_telegram_message(telegram_token, telegram_chat_id, message)
        
        if success:
            print("✅ Segnali inviati con successo a Telegram")
        else:
            print("❌ Invio a Telegram fallito")
    elif all_signals and (not telegram_token or not telegram_chat_id):
        print("ℹ️  Segnali rilevati ma non inviati (credenziali Telegram mancanti)")
    else:
        print("ℹ️  Nessun segnale da inviare")
    
    print(f"\n🏁 Elaborazione completata - {datetime.now().strftime('%H:%M:%S')}")


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

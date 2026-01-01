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
    "PIRC.MI": "Pirelli S.p.A.",
    "RACE.MI": "Ferrari"
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
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            print(f"⚠️  {ticker}: Nessun dato disponibile")
            return None
        
        # yfinance restituisce un DataFrame MultiIndex
        # Convertiamo in DataFrame normale se necessario
        if isinstance(df.columns, pd.MultiIndex):
            df = df[['Open', 'High', 'Low', 'Close']]
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
            
        # Calcola Heikin Ashi Close (ultima candela)
        ha_close = (df['Open'].iloc[-1] + df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 4
        
        # Calcola Heikin Ashi Open (candela precedente)
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
        
        # Calcola massimo e minimo del periodo recente
        recent_max = float(recent_prices.max())
        recent_min = float(recent_prices.min())
        
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


def detect_ma_ema_crossover(df: pd.DataFrame) -> Optional[str]:
    """
    Rileva incroci tra MA31 (rossa) ed EMA10 (verde).
    
    Regole:
    - UP: EMA10 > MA31 (verde)
    - DOWN: MA31 > EMA10 (rossa)
    
    Ritorna:
    - "UP" se crossover rialzista (EMA passa sopra MA)
    - "DOWN" se crossover ribassista (MA passa sopra EMA)
    - "NO_CROSS" se non c'è incrocio
    - None in caso di errore
    """
    try:
        close_prices = df['Close']
        
        if len(close_prices) < 32:  # Abbiamo bisogno di almeno 32 giorni per MA31
            return None
        
        # Calcola MA31 (rossa) e EMA10 (verde)
        ma31 = ta.trend.sma_indicator(close_prices, window=31)
        ema10 = ta.trend.ema_indicator(close_prices, window=10)
        
        # Prendi gli ultimi 2 valori per rilevare il crossover
        if len(ma31) < 2 or len(ema10) < 2:
            return None
        
        # Valori attuali
        ma31_current = float(ma31.iloc[-1])
        ema10_current = float(ema10.iloc[-1])
        
        # Valori del giorno precedente
        ma31_prev = float(ma31.iloc[-2])
        ema10_prev = float(ema10.iloc[-2])
        
        # Rileva crossover
        # Crossover UP: EMA passa sopra MA
        if ema10_prev <= ma31_prev and ema10_current > ma31_current:
            return "UP"
        
        # Crossover DOWN: MA passa sopra EMA
        elif ma31_prev <= ema10_prev and ma31_current > ema10_current:
            return "DOWN"
        
        # Se non c'è incrocio, mostra solo la posizione attuale
        # Ma solo se è cambiata rispetto al giorno prima
        if ema10_current > ma31_current and ema10_prev <= ma31_prev:
            return "UP"
        elif ma31_current > ema10_current and ma31_prev <= ema10_prev:
            return "DOWN"
        
        return "NO_CROSS"
        
    except Exception as e:
        print(f"⚠️  Errore nel rilevamento crossover MA/EMA: {e}")
        return None


def get_ma_ema_position(df: pd.DataFrame) -> Optional[str]:
    """
    Ottiene la posizione attuale tra MA31 e EMA10.
    
    Ritorna:
    - "UP" se EMA10 > MA31 (verde)
    - "DOWN" se MA31 > EMA10 (rossa)
    - None in caso di errore
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


def analyze_ticker_signals(ticker: str) -> List[str]:
    """
    Analizza un singolo ticker e restituisce i segnali rilevati.
    """
    signals = []
    
    # Scarica dati
    df = download_ticker_data(ticker)
    if df is None or len(df) < MIN_DATA_POINTS:
        return signals
    
    try:
        # 1. Analisi crossover MA31/EMA10
        crossover_signal = detect_ma_ema_crossover(df)
        ma_ema_position = get_ma_ema_position(df)
        
        # Se c'è un crossover, lo segnaliamo
        if crossover_signal == "UP":
            signals.append("📈 CROSSOVER UP: EMA10 > MA31")
        elif crossover_signal == "DOWN":
            signals.append("📉 CROSSOVER DOWN: MA31 > EMA10")
        
        # Mostriamo sempre la posizione attuale (anche se non c'è crossover)
        if ma_ema_position == "UP":
            signals.append("🟢 EMA10 > MA31 (Verde)")
        elif ma_ema_position == "DOWN":
            signals.append("🔴 MA31 > EMA10 (Rossa)")
        
        # 2. Analisi direzione ZigZag
        zz_dir = analyze_zigzag_direction(df)
        if zz_dir != "FLAT":
            signals.append(f"📊 ZigZag: {zz_dir}")
        
        # 3. Analisi Heikin Ashi (cambio colore)
        if len(df) > 2:
            color_current = calculate_heikin_ashi_color(df.iloc[-2:])
            color_previous = calculate_heikin_ashi_color(df.iloc[-3:-1])
            
            if color_current != color_previous and color_current != "UNKNOWN":
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
    print(f"Python: {sys.version}")
    
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
    tickers_with_signals = 0
    
    for ticker in portfolio:
        signals = analyze_ticker_signals(ticker)
        
        if signals:
            all_signals[ticker] = signals
            tickers_with_signals += 1
            
            # Log nel terminale
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
            print(f"{time_str} 📬 {ticker}: {' | '.join(signals)}")
    
    # Riepilogo
    total_signals = sum(len(s) for s in all_signals.values())
    print(f"\n📊 Riepilogo:")
    print(f"   Titoli analizzati: {len(portfolio)}")
    print(f"   Titoli con segnali: {tickers_with_signals}")
    print(f"   Segnali totali: {total_signals}")
    
    # Invia a Telegram se ci sono segnali e le credenziali sono valide
    if all_signals and telegram_token and telegram_chat_id:
        print("\n📤 Invio segnali a Telegram...")
        
        message = format_signals_for_telegram(all_signals)
        success = send_telegram_message(telegram_token, telegram_chat_id, message)
        
        if success:
            print(f"✅ Messaggio inviato a Telegram: {tickers_with_signals} titoli, {total_signals} segnali")
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

